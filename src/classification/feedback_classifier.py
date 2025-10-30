#!/usr/bin/env python3
"""
Feedback Classification Service

This module provides feedback classification using a multi-task transformer model
that predicts both Level 1 (Technical, Payment, Claims) and Level 2 (8 subcategories).
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from src.utils.logger import logger


class MultiTaskFeedbackClassifier(nn.Module):
    """
    Multi-task model with shared transformer base and two classification heads

    Architecture:
    Input Text
      ↓
    XLM-RoBERTa (shared)
      ↓
    Pooled Output
      ↓
    ┌─────────────┬─────────────┐
    ↓             ↓             ↓
    Head 1        Head 2
    (Level 1)     (Level 2)
    3 classes     8 classes
    """

    def __init__(
        self,
        model_name: str,
        num_labels_l1: int,
        num_labels_l2: int,
        dropout: float = 0.1,
    ):
        super(MultiTaskFeedbackClassifier, self).__init__()

        # Shared transformer base
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Classification head for Level 1 (3 classes)
        self.classifier_l1 = nn.Linear(hidden_size, num_labels_l1)

        # Classification head for Level 2 (8 classes)
        self.classifier_l2 = nn.Linear(hidden_size, num_labels_l2)

    def forward(self, input_ids, attention_mask, labels_l1=None, labels_l2=None):
        """
        Forward pass

        Returns:
            Dictionary with:
            - loss: Combined loss (if labels provided)
            - logits_l1: Level 1 predictions
            - logits_l2: Level 2 predictions
        """
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Get logits for both tasks
        logits_l1 = self.classifier_l1(pooled_output)
        logits_l2 = self.classifier_l2(pooled_output)

        # Calculate loss if labels provided
        loss = None
        if labels_l1 is not None and labels_l2 is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_l1 = loss_fct(logits_l1, labels_l1)
            loss_l2 = loss_fct(logits_l2, labels_l2)

            # Combined loss (equal weight to both tasks)
            loss = loss_l1 + loss_l2

        return {"loss": loss, "logits_l1": logits_l1, "logits_l2": logits_l2}


class FeedbackClassifier:
    """
    Feedback classification service using a trained multi-task model
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        label_mappings_path: Optional[str] = None,
    ):
        """
        Initialize the feedback classifier

        Args:
            model_path: Path to the trained model directory
            label_mappings_path: Path to the label mappings JSON file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.label_mappings = None
        self.max_length = 128
        self.model_name = "xlm-roberta-base"

        # Set default paths if not provided
        if model_path is None:
            # Try transformer_model directory (primary location)
            transformer_model = (
                Path(__file__).parent.parent.parent
                / "models"
                / "transformer_model"
                / "best_model"
            )
            # Try notebooks directory (training location)
            notebooks_model = (
                Path(__file__).parent.parent.parent
                / "notebooks"
                / "models"
                / "multitask"
                / "best_model"
            )
            # Try legacy path
            legacy_model = (
                Path(__file__).parent.parent.parent
                / "models"
                / "multitask"
                / "best_model"
            )

            if transformer_model.exists():
                model_path = str(transformer_model)
            elif notebooks_model.exists():
                model_path = str(notebooks_model)
            elif legacy_model.exists():
                model_path = str(legacy_model)
            else:
                model_path = None

        if label_mappings_path is None:
            # Try transformer_model directory (primary location)
            transformer_mappings = (
                Path(__file__).parent.parent.parent
                / "models"
                / "transformer_model"
                / "label_mappings_multitask.json"
            )
            # Try notebooks directory
            notebooks_mappings = (
                Path(__file__).parent.parent.parent
                / "notebooks"
                / "label_mappings_multitask.json"
            )
            # Try project root
            project_mappings = (
                Path(__file__).parent.parent.parent / "label_mappings_multitask.json"
            )

            if transformer_mappings.exists():
                label_mappings_path = str(transformer_mappings)
            elif notebooks_mappings.exists():
                label_mappings_path = str(notebooks_mappings)
            elif project_mappings.exists():
                label_mappings_path = str(project_mappings)
            else:
                label_mappings_path = None

        self.model_path = model_path
        self.label_mappings_path = label_mappings_path

        logger.info(f"FeedbackClassifier initialized with device: {self.device}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Label mappings path: {self.label_mappings_path}")

    def load_model(self):
        """
        Load the trained model and label mappings
        """
        if self.model is not None:
            logger.info("Model already loaded")
            return

        # Check if model path exists
        if self.model_path is None or not Path(self.model_path).exists():
            error_msg = f"Model not found at {self.model_path}. Please train the model first using the feedback_classification_multitask.ipynb notebook."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Check if label mappings exist
        if (
            self.label_mappings_path is None
            or not Path(self.label_mappings_path).exists()
        ):
            error_msg = f"Label mappings not found at {self.label_mappings_path}. Please train the model first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Load label mappings
            logger.info(f"Loading label mappings from {self.label_mappings_path}")
            with open(self.label_mappings_path, "r") as f:
                self.label_mappings = json.load(f)

            # Convert string keys to int for id2label
            self.id2label_l1 = {
                int(k): v for k, v in self.label_mappings["level1"]["id2label"].items()
            }
            self.id2label_l2 = {
                int(k): v for k, v in self.label_mappings["level2"]["id2label"].items()
            }

            num_labels_l1 = len(self.id2label_l1)
            num_labels_l2 = len(self.id2label_l2)

            logger.info(
                f"Loaded label mappings: L1={num_labels_l1} classes, L2={num_labels_l2} classes"
            )

            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Initialize model architecture
            logger.info("Initializing model architecture")
            self.model = MultiTaskFeedbackClassifier(
                model_name=self.model_name,
                num_labels_l1=num_labels_l1,
                num_labels_l2=num_labels_l2,
                dropout=0.1,
            )

            # Load trained weights
            logger.info(f"Loading model weights from {self.model_path}")

            # Check for different model file formats
            safetensors_path = Path(self.model_path) / "model.safetensors"
            pytorch_bin_path = Path(self.model_path) / "pytorch_model.bin"

            if safetensors_path.exists():
                # Load from safetensors format
                logger.info("Loading from model.safetensors")
                try:
                    from safetensors.torch import load_file

                    state_dict = load_file(safetensors_path)
                    self.model.load_state_dict(state_dict)
                except ImportError:
                    logger.error(
                        "safetensors package not installed. Please install: pip install safetensors"
                    )
                    raise ImportError(
                        "safetensors package required to load this model format. Install with: pip install safetensors"
                    )
            elif pytorch_bin_path.exists():
                # Load from pytorch binary format
                logger.info("Loading from pytorch_model.bin")
                state_dict = torch.load(
                    pytorch_bin_path, map_location=self.device, weights_only=True
                )
                self.model.load_state_dict(state_dict)
            else:
                # Try loading as pretrained model directory
                logger.warning(
                    "No model.safetensors or pytorch_model.bin found, attempting to load as pretrained model"
                )
                self.model = self.model.from_pretrained(self.model_path)

            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            logger.info("✓ Model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def preprocess_text(self, text: str, lang: str = "en") -> str:
        """
        Minimal preprocessing for transformers

        Args:
            text: Input text
            lang: Language code ('en' or 'ar')

        Returns:
            Preprocessed text
        """
        if pd.isna(text) or not text:
            return ""

        text = str(text)

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove mentions
        text = re.sub(r"@\w+", "", text)

        # Remove hashtags (keep the text)
        text = re.sub(r"#(\w+)", r"\1", text)

        # Arabic-specific preprocessing
        if lang == "ar":
            # Normalize alef variants
            text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            # Remove diacritics
            text = re.sub(r"[ًٌٍَُِّْ]", "", text)

        # Remove extra whitespace
        return " ".join(text.split()).strip()

    def classify(self, text: str, language: str = "en") -> Dict:
        """
        Classify a single feedback text using the multi-task model

        Args:
            text: Feedback text
            language: 'en' or 'ar'

        Returns:
            Dictionary with predictions and probabilities for both levels
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        # Preprocess
        cleaned = self.preprocess_text(text, language)

        if not cleaned:
            raise ValueError("Text is empty after preprocessing")

        # Tokenize
        inputs = self.tokenizer(
            cleaned,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )

        # Get probabilities
        probs_l1 = torch.softmax(outputs["logits_l1"], dim=1)[0]
        probs_l2 = torch.softmax(outputs["logits_l2"], dim=1)[0]

        # Get predictions
        pred_l1 = torch.argmax(probs_l1).item()
        pred_l2 = torch.argmax(probs_l2).item()

        return {
            "text": text,
            "language": language,
            "level1": self.id2label_l1[pred_l1],
            "level1_confidence": probs_l1[pred_l1].item(),
            "level2": self.id2label_l2[pred_l2],
            "level2_confidence": probs_l2[pred_l2].item(),
        }

    def is_ready(self) -> bool:
        """
        Check if the classifier is ready to use

        Returns:
            True if model is loaded and ready
        """
        return self.model is not None

    def get_status(self) -> Dict:
        """
        Get the status of the classifier

        Returns:
            Dictionary with status information
        """
        return {
            "ready": self.is_ready(),
            "model_path": str(self.model_path) if self.model_path else None,
            "label_mappings_path": str(self.label_mappings_path)
            if self.label_mappings_path
            else None,
            "device": str(self.device),
            "model_loaded": self.model is not None,
        }


# Global classifier instance
feedback_classifier = FeedbackClassifier()
