import torch
import torch.nn as nn
import torch.fft
import math

class HybridEmbedding(nn.Module):
    """
    Projette les 55 features hétérogènes (Continues et Catégorielles)
    dans un espace vectoriel unique de dimension d_model.
    """
    def __init__(self, num_cont_features, cat_vocab_sizes, d_model=256):
        super().__init__()
        self.d_model = d_model
        
        # Couche Linéaire pour les variables continues (déjà normalisées en Z-Score)
        self.cont_proj = nn.Linear(num_cont_features, d_model)
        
        # Embeddings pour les variables catégorielles (ex: Protocol, Ports)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
            for vocab_size in cat_vocab_sizes
        ])
        
        # Fusion des embeddings
        self.fusion_layer = nn.Linear(d_model * (1 + len(cat_vocab_sizes)), d_model)

    def forward(self, cont_data, cat_data):
        # 1. Projection Continue: [Batch, Seq, d_model]
        x_cont = self.cont_proj(cont_data)
        
        # 2. Projection Catégorielle: Liste de [Batch, Seq, d_model]
        x_cats = [emb(cat_data[:, :, i]) for i, emb in enumerate(self.cat_embeddings)]
        
        # 3. Concaténation spatiale et Fusion
        x_fused = torch.cat([x_cont] + x_cats, dim=-1)
        return self.fusion_layer(x_fused)

class FrequencyMaskingLayer(nn.Module):
    """
    Implémente le MFM (Masked Frequency Modeling) via RFFT.
    Masque une partie des composantes fréquentielles pour forcer l'apprentissage rythmique (Botnets).
    """
    # MODIFICATION : Paramètre dynamique au lieu du 0.40 en dur
    def __init__(self, mask_ratio=0.10): 
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        # x shape: [Batch, Seq_Len, d_model]
        
        # 1. Passage dans le domaine spectral (Axe Temporel dim=1)
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')
        
        # 2. Génération du masque binaire (True = Masqué)
        batch, freq_len, d_model = x_freq.shape
        mask = torch.rand(batch, freq_len, d_model, device=x.device) < self.mask_ratio
        
        # 3. Application du masque (Mise à zéro des fréquences)
        x_masked_freq = x_freq.clone()
        x_masked_freq[mask] = 0.0
        
        # 4. Retour au domaine temporel
        x_reconstructed_time = torch.fft.irfft(x_masked_freq, n=x.shape[1], dim=1, norm='ortho')
        return x_reconstructed_time

class SpatioTemporalTransformer(nn.Module):
    """
    Le Foundation Model complet : STT + MAE + MFM
    """
    # MODIFICATION : Ajout des ratios initiaux pour le Curriculum Learning
    def __init__(self, num_cont_features, cat_vocab_sizes, seq_len=32, d_model=256, n_heads=8, n_layers=4, init_mae=0.30, init_mfm=0.10):
        super().__init__()
        
        # 1. Tokenizer Spatial
        self.embedding = HybridEmbedding(num_cont_features, cat_vocab_sizes, d_model)
        
        # 2. Positional Encoding (Contexte Temporel)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # 3. Couche MFM (Démarre à init_mfm au lieu de 0.40)
        self.mfm_layer = FrequencyMaskingLayer(mask_ratio=init_mfm)
        
        # 4. STT Encoder (Backbone Lourd)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model*4, 
            dropout=0.1, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 5. Token MAE (Démarre à init_mae au lieu de 0.70)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mae_mask_ratio = init_mae
        
        
        # 6. Décodeur Asymétrique Léger (Pour la Reconstruction Zero-Day)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_cont_features) # On reconstruit les valeurs continues pour calculer la MSE
        )

    def forward(self, cont_data, cat_data):
        batch_size, seq_len, _ = cont_data.shape
        
        # Étape 1 : Embedding Hybride
        x = self.embedding(cont_data, cat_data)
        
       # Étape 2 : Masquage Spatial (MAE dynamique)
        spatial_mask = torch.rand(batch_size, seq_len, device=x.device) < self.mae_mask_ratio
        # MODIFICATION : Cast dynamique du token pour la compatibilité FP16 (autocast)
        x[spatial_mask] = self.mask_token.to(dtype=x.dtype)
        
        
        # Étape 3 : Addition de la position temporelle
        x = x + self.pos_encoder
        
        # Étape 4 : Masquage Spectral (MFM dynamique via FFT)
        x = self.mfm_layer(x)
        
        # Étape 5 : Spatio-Temporal Attention (Le Cerveau)
        encoded_features = self.encoder(x)
        
        # Étape 6 : Reconstruction Asymétrique
        reconstructed_cont = self.decoder(encoded_features)
        
        # On retourne les features reconstruites et le masque spatial 
        return reconstructed_cont, spatial_mask

# ==========================================
# Test d'intégration locale (Compilation)
# ==========================================
if __name__ == "__main__":
    # Simulation des données issues du DataLoader
    BATCH_SIZE, SEQ_LEN = 32, 32
    NUM_CONT = 51 # Exemple : 55 total - 4 catégorielles
    CAT_VOCABS = [10, 65535, 12, 65535] # Protocol, Dst_Port, TCP_Flags, Src_Port
    
    # On initialise avec les valeurs de Warmup
    model = SpatioTemporalTransformer(
        num_cont_features=NUM_CONT, 
        cat_vocab_sizes=CAT_VOCABS,
        init_mae=0.30,
        init_mfm=0.10
    )
    
    
    dummy_cont = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_CONT)
    dummy_cat = torch.randint(0, 10, (BATCH_SIZE, SEQ_LEN, 4))
    
    # Vérification de l'empreinte mémoire initiale
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle STT instancié. Paramètres totaux : {model_params:,}")
    print(f"Ratios initiaux -> MAE: {model.mae_mask_ratio} | MFM: {model.mfm_layer.mask_ratio}")
    
    # Test du Forward Pass
    reconstruction, mask = model(dummy_cont, dummy_cat)
    print(f"Shape de sortie (Reconstruction) : {reconstruction.shape} (Attendu: [{BATCH_SIZE}, {SEQ_LEN}, {NUM_CONT}])")
    print("✅ Forward Pass validé. Architecture prête pour l'entraînement avec Curriculum Learning.")