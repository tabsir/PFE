import torch
import torch.nn as nn
import torch.fft
import math

class HybridEmbedding(nn.Module):
    def __init__(self, num_cont_features, cat_vocab_sizes, d_model=256):
        super().__init__()
        self.d_model = d_model
        # Projection des variables continues (40)
        self.cont_proj = nn.Linear(num_cont_features, d_model)
        
        # Embeddings pour les 9 variables catégorielles
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
            for vocab_size in cat_vocab_sizes
        ])
        
        # Fusion : d_model * (1 projection continue + 9 catégories)
        self.fusion_layer = nn.Linear(d_model * (1 + len(cat_vocab_sizes)), d_model)

    def forward(self, cont_data, cat_data):
        x_cont = self.cont_proj(cont_data) # [Batch, Seq, d_model]
        
        # Embeddings catégoriels
        x_cats = [emb(cat_data[:, :, i]) for i, emb in enumerate(self.cat_embeddings)]
        
        # Concaténation et réduction vers d_model
        x_fused = torch.cat([x_cont] + x_cats, dim=-1)
        return self.fusion_layer(x_fused)

class FrequencyMaskingLayer(nn.Module):
    def __init__(self, mask_ratio=0.10):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        # Domaine spectral
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')
        batch, freq_len, d_model = x_freq.shape
        
        mask = torch.rand(batch, freq_len, d_model, device=x.device) < self.mask_ratio
        
        x_masked_freq = x_freq.clone()
        x_masked_freq[mask] = 0.0
        
        return torch.fft.irfft(x_masked_freq, n=x.shape[1], dim=1, norm='ortho')

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_cont_features, cat_vocab_sizes, seq_len=32, d_model=256, n_heads=8, n_layers=4, init_mae=0.30, init_mfm=0.10):
        super().__init__()
        
        self.embedding = HybridEmbedding(num_cont_features, cat_vocab_sizes, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.mfm_layer = FrequencyMaskingLayer(mask_ratio=init_mfm)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mae_mask_ratio = init_mae
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_cont_features)
        )

    def forward(self, cont_data, cat_data):
        batch_size, seq_len, _ = cont_data.shape
        
        x = self.embedding(cont_data, cat_data)
        
        # MAE (Masked Auto-Encoder) spatial
        spatial_mask = torch.rand(batch_size, seq_len, device=x.device) < self.mae_mask_ratio
        x[spatial_mask] = self.mask_token.to(dtype=x.dtype)
        
        x = x + self.pos_encoder
        x = self.mfm_layer(x)
        
        encoded_features = self.encoder(x)
        reconstructed_cont = self.decoder(encoded_features)
        
        return reconstructed_cont, spatial_mask

# ==========================================
# Test d'intégration locale (Corrigé)
# ==========================================
if __name__ == "__main__":
    # Paramètres synchronisés avec NF-v3
    BATCH_SIZE, SEQ_LEN = 128, 32
    NUM_CONT = 40  # <--- Correction ici (ton JSON a 40 variables)
    
    # Vocabulaires pour tes 9 catégories
    # On met 65536 pour les ports et 256 pour les protocoles/flags pour être safe
    CAT_VOCABS = [256, 256, 256, 65536, 65536, 256, 256, 256, 256] 

    model = SpatioTemporalTransformer(
        num_cont_features=NUM_CONT,
        cat_vocab_sizes=CAT_VOCABS,
        init_mae=0.30,
        init_mfm=0.10
    )

    dummy_cont = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_CONT)
    dummy_cat = torch.randint(0, 255, (BATCH_SIZE, SEQ_LEN, 9)) # <--- 9 catégories

    model_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle STT instancié. Paramètres totaux : {model_params:,}")

    reconstruction, mask = model(dummy_cont, dummy_cat)
    print(f"Shape de sortie (Reconstruction) : {reconstruction.shape}")
    print("✅ Forward Pass validé.")