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
        
        # Embeddings catégoriels avec SÉCURITÉ CUDA (Bouclier Anti-Crash)
        x_cats = []
        for i, emb in enumerate(self.cat_embeddings):
            # torch.clamp force l'index mathématiquement entre 0 et vocab_size - 1
            # Les nombres négatifs (NaN) deviennent 0 (Unknown)
            # Les nombres trop grands (ex: 280) sont plafonnés au maximum autorisé
            safe_cat = torch.clamp(cat_data[:, :, i], min=0, max=emb.num_embeddings - 1)
            x_cats.append(emb(safe_cat))
        
        # Concaténation et réduction vers d_model
        x_fused = torch.cat([x_cont] + x_cats, dim=-1)
        return self.fusion_layer(x_fused)

class FrequencyMaskingLayer(nn.Module):
    def __init__(self, mask_ratio=0.10):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x, apply_mask=None):
        if apply_mask is None:
            apply_mask = self.training

        if not apply_mask or self.mask_ratio <= 0:
            return x

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
        self.d_model = d_model
        
        self.embedding = HybridEmbedding(num_cont_features, cat_vocab_sizes, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.mfm_layer = FrequencyMaskingLayer(mask_ratio=init_mfm)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True, norm_first=True  # Pre-LN for stable gradients
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model)  # Final LayerNorm prevents gradient vanishing
        )
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)  # Small random init, not zeros
        self.mae_mask_ratio = init_mae
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_cont_features)
        )

    def encode_features(self, cont_data, cat_data, spatial_mask=None, apply_mfm=None):
        batch_size, seq_len, _ = cont_data.shape

        x = self.embedding(cont_data, cat_data)

        # MAE (Masked Auto-Encoder) spatial — explicit mask support for validation without dropout.
        if spatial_mask is None and self.training:
            spatial_mask = torch.rand(batch_size, seq_len, device=x.device) < self.mae_mask_ratio
        elif spatial_mask is None:
            spatial_mask = torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.bool)
        else:
            spatial_mask = spatial_mask.to(device=x.device, dtype=torch.bool)

        if spatial_mask.any():
            x = x.clone()
            x[spatial_mask] = self.mask_token[0, 0].to(dtype=x.dtype)

        x = x + self.pos_encoder[:, :seq_len]
        x = self.mfm_layer(x, apply_mask=apply_mfm)
        encoded_features = self.encoder(x)
        return encoded_features, spatial_mask

    def forward(self, cont_data, cat_data, spatial_mask=None, apply_mfm=None):
        encoded_features, spatial_mask = self.encode_features(
            cont_data,
            cat_data,
            spatial_mask=spatial_mask,
            apply_mfm=apply_mfm,
        )
        reconstructed_cont = self.decoder(encoded_features)
        
        return reconstructed_cont, spatial_mask


class NIDSMultiTaskModel(nn.Module):
    def __init__(self, backbone, num_known_attack_classes, dropout=0.10):
        super().__init__()
        self.backbone = backbone
        self.num_known_attack_classes = num_known_attack_classes

        pooled_dim = backbone.d_model * 2
        self.pool_norm = nn.LayerNorm(pooled_dim)
        self.shared_projection = nn.Sequential(
            nn.Linear(pooled_dim, backbone.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.current_attack_head = nn.Linear(backbone.d_model, 1)
        self.future_attack_head = nn.Linear(backbone.d_model, 1)
        self.attack_family_head = (
            nn.Linear(backbone.d_model, num_known_attack_classes)
            if num_known_attack_classes > 0
            else None
        )

    def pool_encoded_features(self, encoded_features):
        mean_pool = encoded_features.mean(dim=1)
        max_pool = encoded_features.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        pooled = self.pool_norm(pooled)
        return self.shared_projection(pooled)

    def forward(self, cont_data, cat_data, apply_mfm=False):
        spatial_mask = torch.zeros(
            cont_data.shape[0],
            cont_data.shape[1],
            device=cont_data.device,
            dtype=torch.bool,
        )
        encoded_features, _ = self.backbone.encode_features(
            cont_data,
            cat_data,
            spatial_mask=spatial_mask,
            apply_mfm=apply_mfm,
        )
        pooled_features = self.pool_encoded_features(encoded_features)

        outputs = {
            'current_attack_logits': self.current_attack_head(pooled_features).squeeze(-1),
            'future_attack_logits': self.future_attack_head(pooled_features).squeeze(-1),
            'pooled_features': pooled_features,
        }

        if self.attack_family_head is not None:
            outputs['attack_family_logits'] = self.attack_family_head(pooled_features)
        else:
            outputs['attack_family_logits'] = None

        return outputs

    @staticmethod
    def decode_predictions(outputs, attack_labels, current_threshold=0.50, known_attack_threshold=0.55, future_threshold=0.50):
        current_probs = torch.sigmoid(outputs['current_attack_logits']).detach().cpu()
        future_probs = torch.sigmoid(outputs['future_attack_logits']).detach().cpu()

        if outputs.get('attack_family_logits') is not None:
            family_probs = torch.softmax(outputs['attack_family_logits'], dim=-1).detach().cpu()
            known_confidence, known_index = family_probs.max(dim=-1)
        else:
            batch_size = current_probs.shape[0]
            family_probs = None
            known_confidence = torch.zeros(batch_size)
            known_index = torch.zeros(batch_size, dtype=torch.long)

        decoded = []
        for idx in range(current_probs.shape[0]):
            current_prob = float(current_probs[idx])
            future_prob = float(future_probs[idx])
            known_conf = float(known_confidence[idx])
            sample = {
                'current_attack_probability': current_prob,
                'future_attack_probability': future_prob,
                'future_warning': future_prob >= future_threshold,
            }

            if current_prob < current_threshold:
                sample['status'] = 'benign'
                sample['attack_type'] = 'Benign'
            elif family_probs is not None and known_conf >= known_attack_threshold and attack_labels:
                attack_idx = int(known_index[idx])
                sample['status'] = 'known_attack'
                sample['attack_type'] = attack_labels[attack_idx]
                sample['known_attack_confidence'] = known_conf
            else:
                sample['status'] = 'unknown_attack_warning'
                sample['attack_type'] = 'Unknown'
                sample['known_attack_confidence'] = known_conf

            decoded.append(sample)

        return decoded

# ==========================================
# Test d'intégration locale (Corrigé)
# ==========================================
if __name__ == "__main__":
    # Paramètres synchronisés avec NF-v3
    BATCH_SIZE, SEQ_LEN = 512, 32
    NUM_CONT = 40  # <--- Correction ci (ton JSON a 40 variables)
    
    # Vocabulaires pour tes 9 catégories
    # On met 65536 pour les ports et 256 pour les protocoles/flags pour être safe
    CAT_VOCABS = [256, 256, 256, 65536, 65536, 256, 256, 256, 256] 

    model = SpatioTemporalTransformer(
        num_cont_features=NUM_CONT,
        cat_vocab_sizes=CAT_VOCABS,
        init_mae=0.30,
        init_mfm=0.00
    )

    dummy_cont = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_CONT)
    dummy_cat = torch.randint(0, 255, (BATCH_SIZE, SEQ_LEN, 9)) # <--- 9 catégories

    model_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle STT instancié. Paramètres totaux : {model_params:,}")

    reconstruction, mask = model(dummy_cont, dummy_cat)
    print(f"Shape de sortie (Reconstruction) : {reconstruction.shape}")
    print("✅ Forward Pass validé.")