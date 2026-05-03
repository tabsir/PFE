import math

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


def build_fixed_spatial_mask(batch_size, seq_len, mask_ratio, device, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return (torch.rand(batch_size, seq_len, generator=generator) < float(mask_ratio)).to(device)


def compute_reconstruction_metrics(reconstructed, target, spatial_mask):
    mse_raw = (reconstructed - target).pow(2)
    robust_raw = F.smooth_l1_loss(reconstructed, target, reduction="none", beta=1.0)

    mask_expanded = spatial_mask.unsqueeze(-1).float()
    masked_denominator = mask_expanded.sum().clamp(min=1.0) * reconstructed.shape[-1]

    masked_robust = (robust_raw * mask_expanded).sum() / masked_denominator
    full_robust = robust_raw.mean()
    masked_mse = (mse_raw * mask_expanded).sum() / masked_denominator
    full_mse = mse_raw.mean()

    return {
        "train_loss": 0.85 * masked_robust + 0.15 * full_robust,
        "masked_mse": masked_mse,
        "full_mse": full_mse,
    }


def build_reconstruction_calibration(scores, labels, quantile_count=101):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    benign_scores = scores[labels == 0]
    benign_scores = benign_scores[np.isfinite(benign_scores)]
    if benign_scores.size == 0:
        return None

    quantile_levels = np.linspace(0.0, 1.0, int(quantile_count))
    quantile_values = np.quantile(benign_scores, quantile_levels)
    return {
        "quantile_levels": quantile_levels.tolist(),
        "quantile_values": quantile_values.tolist(),
        "benign_mean": float(benign_scores.mean()),
        "benign_std": float(benign_scores.std()),
        "score_name": "masked_reconstruction_mse",
        "normalization": "empirical_benign_percentile",
    }


def reconstruction_scores_to_percentiles(scores, calibration):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if calibration is None:
        return np.zeros_like(scores, dtype=np.float64)

    quantile_levels = np.asarray(calibration.get("quantile_levels", []), dtype=np.float64)
    quantile_values = np.asarray(calibration.get("quantile_values", []), dtype=np.float64)
    if quantile_levels.size == 0 or quantile_values.size == 0:
        return np.zeros_like(scores, dtype=np.float64)

    unique_values, unique_indices = np.unique(quantile_values, return_index=True)
    unique_levels = quantile_levels[unique_indices]
    if unique_values.size == 1:
        return np.where(scores >= unique_values[0], 1.0, 0.0)
    return np.interp(scores, unique_values, unique_levels, left=0.0, right=1.0)


def combine_reconstruction_scores(mae_scores, mfm_scores):
    mae_scores = np.asarray(mae_scores, dtype=np.float64).reshape(-1)
    mfm_scores = np.asarray(mfm_scores, dtype=np.float64).reshape(-1)
    if mae_scores.size == 0:
        return mfm_scores
    if mfm_scores.size == 0:
        return mae_scores
    return mae_scores + mfm_scores


def combine_unknown_scores(raw_unknown_probabilities, reconstruction_probabilities):
    raw_unknown_probabilities = np.asarray(raw_unknown_probabilities, dtype=np.float64).reshape(-1)
    reconstruction_probabilities = np.asarray(reconstruction_probabilities, dtype=np.float64).reshape(-1)
    if raw_unknown_probabilities.size == 0:
        return reconstruction_probabilities
    if reconstruction_probabilities.size == 0:
        return raw_unknown_probabilities
    return np.maximum(raw_unknown_probabilities, reconstruction_probabilities)


def compute_combined_reconstruction_outputs(
    model,
    cont_data,
    cat_data,
    mae_reconstruction_mask,
    mfm_reconstruction_mask=None,
):
    outputs = model(
        cont_data,
        cat_data,
        apply_mfm=False,
        compute_reconstruction=True,
        reconstruction_mask=mae_reconstruction_mask,
        reconstruction_apply_mfm=False,
    )

    mae_reconstruction_score = outputs.get("reconstruction_score")
    if mae_reconstruction_score is None:
        batch_size = cont_data.shape[0]
        mae_reconstruction_score = cont_data.new_zeros(batch_size)

    if mfm_reconstruction_mask is None:
        outputs["mae_reconstruction_score"] = mae_reconstruction_score
        outputs["mfm_reconstruction_score"] = cont_data.new_zeros(mae_reconstruction_score.shape)
        outputs["reconstruction_score_mode"] = "mae_only"
        return outputs

    mfm_outputs = model(
        cont_data,
        cat_data,
        apply_mfm=False,
        compute_reconstruction=True,
        reconstruction_mask=mfm_reconstruction_mask,
        reconstruction_apply_mfm=True,
    )
    mfm_reconstruction_score = mfm_outputs.get("reconstruction_score")
    if mfm_reconstruction_score is None:
        mfm_reconstruction_score = mae_reconstruction_score.new_zeros(mae_reconstruction_score.shape)

    combined_scores = combine_reconstruction_scores(
        mae_reconstruction_score.detach().cpu().numpy(),
        mfm_reconstruction_score.detach().cpu().numpy(),
    )
    outputs["mae_reconstruction_score"] = mae_reconstruction_score
    outputs["mfm_reconstruction_score"] = mfm_reconstruction_score
    outputs["reconstruction_score"] = mae_reconstruction_score.new_tensor(combined_scores)
    outputs["reconstruction_score_mode"] = "combined_mae_mfm"
    return outputs

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
    def __init__(
        self,
        backbone,
        num_known_attack_classes,
        dropout=0.10,
        use_future_head=True,
        use_ood_head=True,
        future_horizons_minutes=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_known_attack_classes = num_known_attack_classes
        self.use_future_head = use_future_head
        self.use_ood_head = use_ood_head
        self.future_horizons_minutes = [
            int(value)
            for value in (future_horizons_minutes or ([5] if use_future_head else []))
        ]
        self.num_future_horizons = len(self.future_horizons_minutes) if use_future_head else 0

        pooled_dim = backbone.d_model * 2
        self.pool_norm = nn.LayerNorm(pooled_dim)
        self.shared_projection = nn.Sequential(
            nn.Linear(pooled_dim, backbone.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.current_attack_head = nn.Linear(backbone.d_model, 1)
        self.future_attack_head = (
            nn.Linear(backbone.d_model, self.num_future_horizons)
            if self.num_future_horizons > 0
            else None
        )
        self.unknown_attack_head = nn.Linear(backbone.d_model, 1) if use_ood_head else None
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

    def forward(
        self,
        cont_data,
        cat_data,
        apply_mfm=False,
        compute_reconstruction=False,
        reconstruction_mask=None,
        reconstruction_apply_mfm=None,
    ):
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
            'future_attack_logits': self.future_attack_head(pooled_features)
            if self.future_attack_head is not None
            else None,
            'unknown_attack_logits': self.unknown_attack_head(pooled_features).squeeze(-1)
            if self.unknown_attack_head is not None
            else None,
            'pooled_features': pooled_features,
        }

        if self.attack_family_head is not None:
            outputs['attack_family_logits'] = self.attack_family_head(pooled_features)
        else:
            outputs['attack_family_logits'] = None

        if compute_reconstruction:
            reconstruction_encoded, reconstruction_mask = self.backbone.encode_features(
                cont_data,
                cat_data,
                spatial_mask=reconstruction_mask,
                apply_mfm=reconstruction_apply_mfm,
            )
            reconstructed_cont = self.backbone.decoder(reconstruction_encoded)
            reconstruction_score = (reconstructed_cont - cont_data).pow(2).mean(dim=(1, 2))
            outputs['reconstructed_cont'] = reconstructed_cont
            outputs['reconstruction_mask'] = reconstruction_mask
            outputs['reconstruction_score'] = reconstruction_score
        else:
            outputs['reconstructed_cont'] = None
            outputs['reconstruction_mask'] = None
            outputs['reconstruction_score'] = None

        return outputs

    @staticmethod
    def decode_predictions(
        outputs,
        attack_labels,
        current_threshold=0.50,
        known_attack_threshold=0.55,
        future_threshold=0.50,
        ood_threshold=0.50,
        future_horizons_minutes=None,
        reconstruction_calibration=None,
        hybrid_ood_threshold=None,
    ):
        current_probs = torch.sigmoid(outputs['current_attack_logits']).detach().cpu()
        future_logits = outputs.get('future_attack_logits')
        future_task_enabled = future_logits is not None and future_logits.numel() > 0
        if future_task_enabled:
            future_probs = torch.sigmoid(future_logits).detach().cpu()
            if future_probs.ndim == 1:
                future_probs = future_probs.unsqueeze(-1)
        else:
            future_probs = torch.zeros(current_probs.shape[0], 0)

        if future_horizons_minutes is None:
            future_horizons_minutes = list(range(1, future_probs.shape[-1] + 1))
        else:
            future_horizons_minutes = list(future_horizons_minutes)
        if future_probs.shape[-1] != len(future_horizons_minutes):
            future_horizons_minutes = future_horizons_minutes[: future_probs.shape[-1]]
        future_horizon_labels = [f"{int(value)}m" for value in future_horizons_minutes]
        if not future_horizon_labels:
            future_task_enabled = False

        if isinstance(future_threshold, dict):
            future_thresholds = {
                label: float(
                    future_threshold.get(
                        label,
                        future_threshold.get(str(future_horizons_minutes[idx]), future_threshold.get("future", 0.50)),
                    )
                )
                for idx, label in enumerate(future_horizon_labels)
            }
        else:
            future_thresholds = {label: float(future_threshold) for label in future_horizon_labels}

        ood_task_enabled = outputs.get('unknown_attack_logits') is not None
        if ood_task_enabled:
            unknown_probs = torch.sigmoid(outputs['unknown_attack_logits']).detach().cpu()
        else:
            unknown_probs = torch.zeros_like(current_probs)

        reconstruction_scores = outputs.get('reconstruction_score')
        if reconstruction_scores is not None:
            reconstruction_scores = reconstruction_scores.detach().cpu().numpy()
            reconstruction_probabilities = reconstruction_scores_to_percentiles(
                reconstruction_scores,
                reconstruction_calibration,
            )
        else:
            reconstruction_scores = np.zeros(current_probs.shape[0], dtype=np.float64)
            reconstruction_probabilities = np.zeros(current_probs.shape[0], dtype=np.float64)

        raw_unknown_probabilities = unknown_probs.numpy()
        hybrid_unknown_probabilities = combine_unknown_scores(
            raw_unknown_probabilities,
            reconstruction_probabilities,
        )
        unknown_threshold = float(hybrid_ood_threshold) if hybrid_ood_threshold is not None else float(ood_threshold)

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
            raw_unknown_prob = float(raw_unknown_probabilities[idx])
            reconstruction_prob = float(reconstruction_probabilities[idx])
            reconstruction_score_value = float(reconstruction_scores[idx])
            hybrid_unknown_risk_prob = float(hybrid_unknown_probabilities[idx])
            known_conf = float(known_confidence[idx])
            current_alarm = current_prob >= current_threshold
            novelty_alarm = hybrid_unknown_risk_prob >= unknown_threshold
            future_probabilities = (
                {
                    label: float(future_probs[idx, horizon_idx])
                    for horizon_idx, label in enumerate(future_horizon_labels)
                }
                if future_task_enabled
                else {}
            )
            future_warnings = {
                label: future_probabilities[label] >= future_thresholds[label]
                for label in future_horizon_labels
            }
            future_warning_any = any(future_warnings.values())
            family_rejection = bool(
                current_alarm and family_probs is not None and attack_labels and known_conf < known_attack_threshold
            )
            unknown_attack_warning = bool(current_alarm and (novelty_alarm or family_rejection))
            novelty_watch = bool((not current_alarm) and novelty_alarm)
            sample = {
                'current_attack_probability': current_prob,
                'present_attack_probability': current_prob,
                'future_task_enabled': future_task_enabled,
                'future_horizons_minutes': list(future_horizons_minutes),
                'future_attack_probability': max(future_probabilities.values()) if future_probabilities else 0.0,
                'future_attack_probabilities': future_probabilities,
                'future_warning': future_warning_any,
                'future_warnings_by_horizon': future_warnings,
                'ood_task_enabled': bool(ood_task_enabled or reconstruction_calibration is not None),
                'raw_ood_head_enabled': bool(ood_task_enabled),
                'reconstruction_ood_enabled': reconstruction_calibration is not None,
                'unknown_head_enabled': bool(ood_task_enabled),
                'reconstruction_novelty_enabled': reconstruction_calibration is not None,
                'hybrid_unknown_risk_enabled': bool(ood_task_enabled or reconstruction_calibration is not None),
                'raw_unknown_head_probability': raw_unknown_prob,
                'reconstruction_novelty_score': reconstruction_score_value,
                'reconstruction_novelty_probability': reconstruction_prob,
                'hybrid_unknown_risk_probability': hybrid_unknown_risk_prob,
                'novelty_watch': novelty_watch,
                'unknown_attack_warning': unknown_attack_warning,
                'family_rejection_warning': family_rejection,
                'decision_policy': 'two_stage_current_then_novelty',
                'unknown_risk_threshold': unknown_threshold,
                'current_attack_threshold': float(current_threshold),
                'known_attack_threshold': float(known_attack_threshold),
                'unknown_attack_probability': hybrid_unknown_risk_prob,
                'raw_unknown_attack_probability': raw_unknown_prob,
                'reconstruction_anomaly_score': reconstruction_score_value,
                'reconstruction_anomaly_probability': reconstruction_prob,
            }

            if novelty_watch:
                sample['status'] = 'novelty_watch'
                sample['attack_type'] = 'Novelty Watch'
            elif not current_alarm and not novelty_alarm:
                sample['status'] = 'benign'
                sample['attack_type'] = 'Benign'
            elif (
                current_alarm
                and not novelty_alarm
                and not family_rejection
                and family_probs is not None
                and known_conf >= known_attack_threshold
                and attack_labels
            ):
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