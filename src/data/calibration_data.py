"""
Bundled calibration data and loading utilities.
Provides real text samples for calibration when no external dataset is available.
"""
import json
import os
from typing import List, Optional

from datasets import Dataset  # type: ignore

# 100 bundled calibration samples covering diverse topics
BUNDLED_CALIBRATION = [
    "The quick brown fox jumps over the lazy dog. This is a sample text for calibration purposes.",
    "Artificial intelligence is transforming the world of technology and science.",
    "The Navier-Stokes equations describe the motion of viscous fluid substances.",
    "Transformers have revolutionized natural language processing and computer vision.",
    "Climate change is one of the most critical challenges facing humanity today.",
    "Quantum computing promises to solve problems that are intractable for classical computers.",
    "The human brain contains approximately 86 billion neurons connected by synapses.",
    "Machine learning algorithms learn patterns from data without explicit programming.",
    "The periodic table organizes chemical elements by atomic number and properties.",
    "Space exploration has revealed the vastness and complexity of the universe.",
    "Democracy requires an informed citizenry to function effectively in society.",
    "The printing press revolutionized the spread of knowledge and information.",
    "The industrial revolution transformed society from agrarian to manufacturing economies.",
    "The human genome project mapped all the genes in human DNA successfully.",
    "Plate tectonics explains the movement of Earth's continents over geological time.",
    "The electromagnetic spectrum ranges from radio waves to gamma rays.",
    "Supply and demand are fundamental concepts in economics and market theory.",
    "The water cycle describes how water moves through the environment continuously.",
    "Photosynthesis and cellular respiration are complementary biological processes.",
    "The solar system consists of eight planets orbiting the sun in elliptical paths.",
    "The French Revolution had a profound impact on modern political thought worldwide.",
    "The Renaissance was a period of great cultural and artistic achievement in Europe.",
    "The internet has fundamentally changed how we communicate and access information.",
    "Machine translation systems use neural networks to translate between languages.",
    "Autonomous vehicles use sensors and AI to navigate without human intervention.",
    "Recommendation systems power the personalized content we see on streaming platforms.",
    "Speech recognition technology has improved dramatically with deep learning advances.",
    "Computer vision enables machines to interpret and understand visual information.",
    "Reinforcement learning trains agents to make decisions through trial and error.",
    "Generative adversarial networks can create realistic synthetic images and videos.",
    "Transfer learning allows models to apply knowledge from one task to another.",
    "Federated learning enables training on distributed data without centralizing it.",
    "Edge computing brings computation closer to where data is generated and used.",
    "The metaverse represents a convergence of virtual and physical reality spaces.",
    "Augmented reality overlays digital information onto the real world environment.",
    "Natural language understanding enables computers to comprehend human language.",
    "Knowledge graphs represent relationships between entities in a structured format.",
    "The theory of relativity fundamentally changed our understanding of space and time.",
    "Evolution by natural selection is the cornerstone of modern biology and life sciences.",
    "The water cycle describes how water moves through the environment continuously.",
    "Renewable energy sources like solar and wind power are becoming cost-effective.",
    "The Internet of Things connects billions of devices worldwide for data exchange.",
    "Blockchain technology enables decentralized trust without intermediaries or banks.",
    "The COVID-19 pandemic accelerated the adoption of remote work technologies globally.",
    "Space exploration has led to numerous technological innovations that benefit Earth.",
    "Deep learning has achieved remarkable results in computer vision and NLP tasks.",
    "The future of AI depends on our ability to make models more efficient and accessible.",
    "Neural networks can approximate any continuous function given enough capacity.",
    "The transformer architecture uses self-attention mechanisms for sequence processing.",
    "Machine learning models require large amounts of data to train effectively.",
    "Supervised learning uses labeled data to train predictive models for classification.",
    "Unsupervised learning finds hidden patterns in unlabeled data without guidance.",
    "Semi-supervised learning combines labeled and unlabeled data for better training.",
    "Self-supervised learning creates labels from the data itself for pre-training.",
    "Few-shot learning enables models to generalize from very few examples efficiently.",
    "Meta-learning teaches models how to learn new tasks more quickly and effectively.",
    "Multi-task learning trains a single model to perform multiple related tasks.",
    "Ensemble methods combine multiple models to improve prediction accuracy and robustness.",
    "Decision trees are interpretable models that split data based on feature values.",
    "Random forests combine many decision trees to reduce overfitting and improve accuracy.",
    "Support vector machines find optimal hyperplanes to separate classes in high dimensions.",
    "K-nearest neighbors classify points based on the majority vote of their neighbors.",
    "Principal component analysis reduces dimensionality while preserving data variance.",
    "t-SNE visualizes high-dimensional data in two or three dimensions for exploration.",
    "K-means clustering partitions data into K groups based on distance to centroids.",
    "Hierarchical clustering builds a tree of clusters by merging similar groups iteratively.",
    "DBSCAN finds clusters of arbitrary shape based on density of data points.",
    "Gaussian mixture models represent data as a mixture of multiple Gaussian distributions.",
    "Hidden Markov models are used for sequential data like speech and text processing.",
    "Conditional random fields are used for structured prediction tasks in NLP.",
    "Recurrent neural networks process sequential data by maintaining hidden states.",
    "Long short-term memory networks address the vanishing gradient problem in RNNs.",
    "Gated recurrent units are simplified versions of LSTM with fewer parameters.",
    "Convolutional neural networks use filters to detect spatial patterns in images.",
    "ResNet introduced residual connections that enable training very deep networks.",
    "Inception networks use parallel convolutions of different sizes for multi-scale features.",
    "MobileNet uses depthwise separable convolutions for efficient on-device inference.",
    "EfficientNet optimizes network depth, width, and resolution jointly for efficiency.",
    "Vision transformers apply transformer architecture to image classification tasks.",
    "BERT uses bidirectional context for deep language understanding and representation.",
    "GPT models generate text autoregressively using transformer decoder architecture.",
    "T5 frames all NLP tasks as text-to-text problems for unified modeling.",
    "BART combines bidirectional and autoregressive training for text generation.",
    "XLNet uses permutation language modeling to capture bidirectional context.",
    "RoBERTa optimizes BERT training with more data and longer training duration.",
    "ALBERT reduces memory usage through parameter sharing across transformer layers.",
    "DistilBERT distills knowledge from larger models into smaller efficient versions.",
    "ELECTRA uses discriminative training to detect replaced tokens in text.",
    "Longformer uses sparse attention patterns to process long documents efficiently.",
    "BigBird combines sparse attention with global tokens for very long sequences.",
    "Reformer uses locality-sensitive hashing for efficient attention computation.",
    "Linformer projects attention to lower dimensions for linear complexity.",
    "Performer uses kernel methods to approximate attention with linear complexity.",
    "Sparse transformers use fixed sparse attention patterns for efficiency gains.",
    "Compressive transformers cache and compress past activations for long sequences.",
    "Adaptive computation time allows models to dynamically adjust computation per step.",
    "Universal transformers combine transformers with recurrent inductive biases.",
    "The attention mechanism computes weighted sums of values based on query-key similarities.",
    "Multi-head attention runs multiple attention operations in parallel for diversity.",
    "Self-attention computes attention between all pairs of positions in a sequence.",
    "Cross-attention computes attention between two different sequences for alignment.",
    "Positional encodings inject information about token positions into the model.",
    "Layer normalization stabilizes training by normalizing activations across features.",
    "Dropout randomly masks neurons during training to prevent overfitting and co-adaptation.",
    "Gradient clipping prevents exploding gradients by scaling down large gradients.",
    "Learning rate scheduling adjusts the learning rate during training for better convergence.",
    "Weight decay adds a penalty on large weights to regularize the model during training.",
    "Adam optimizer combines momentum and adaptive learning rates for efficient training.",
]


def get_calibration_data(
    num_samples: int = 100,
    source: str = "bundled",
    file_path: Optional[str] = None,
) -> Dataset:
    """
    Get calibration data from bundled source, file, or HuggingFace datasets.

    Args:
        num_samples: Number of calibration samples to use
        source: "bundled", "file", "huggingface", or "auto"
        file_path: Path to a .txt or .jsonl file (used when source="file")

    Returns:
        HuggingFace Dataset with a "text" column
    """
    texts: List[str] = []

    if source == "bundled" or (source == "auto" and file_path is None):
        # Use bundled calibration data
        texts = BUNDLED_CALIBRATION[:num_samples]
        while len(texts) < num_samples:
            texts.extend(BUNDLED_CALIBRATION)
        texts = texts[:num_samples]

    elif file_path is not None and os.path.exists(file_path):
        # Load from file
        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        texts.append(json.loads(line)["text"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        texts = texts[:num_samples]

    elif source == "huggingface" or source == "auto":
        # Try HuggingFace datasets
        try:
            from datasets import load_dataset  # type: ignore

            dataset = load_dataset("c4", split="train", streaming=True)
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                texts.append(sample["text"])
        except Exception:
            # Final fallback to bundled
            texts = BUNDLED_CALIBRATION[:num_samples]
            while len(texts) < num_samples:
                texts.extend(BUNDLED_CALIBRATION)
            texts = texts[:num_samples]
    else:
        # Fallback to bundled
        texts = BUNDLED_CALIBRATION[:num_samples]
        while len(texts) < num_samples:
            texts.extend(BUNDLED_CALIBRATION)
        texts = texts[:num_samples]

    if not texts:
        texts = BUNDLED_CALIBRATION[:num_samples]

    return Dataset.from_dict({"text": texts})