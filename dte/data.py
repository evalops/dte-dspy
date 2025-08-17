"""
Dataset loading and management for DTE experiments.

This module provides utilities for loading, creating, and managing
datasets for disagreement-triggered escalation experiments.
"""

import json
import random
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import pandas as pd


def create_test_dataset(size: int = 20) -> Tuple[List[str], List[str]]:
    """
    Create a balanced test dataset of factual claims.
    
    Args:
        size: Number of claims to generate (will be made even)
        
    Returns:
        Tuple of (claims, ground_truth_labels)
    """
    # Ensure even size for balanced dataset
    size = size if size % 2 == 0 else size + 1
    half_size = size // 2
    
    true_claims = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Shakespeare wrote Romeo and Juliet.",
        "The Earth orbits around the Sun.",
        "Python is a programming language.",
        "The Pacific Ocean is the largest ocean.",
        "DNA stands for Deoxyribonucleic acid.",
        "The Great Wall of China is in China.",
        "Gravity causes objects to fall.",
        "The human heart has four chambers.",
        "Gold is a chemical element.",
        "The sun is a star.",
        "Humans have two eyes.",
        "The Earth is round.",
        "Antarctica is the coldest continent.",
    ]
    
    false_claims = [
        "The capital of France is London.",
        "Water boils at 50 degrees Celsius at sea level.",
        "Shakespeare wrote Harry Potter.",
        "The Sun orbits around the Earth.",
        "Python is only a type of snake.",
        "The Atlantic Ocean is the largest ocean.",
        "DNA stands for Digital Network Access.",
        "The Great Wall of China is in Japan.",
        "Gravity causes objects to float.",
        "The human heart has two chambers.",
        "Gold is made of plastic.",
        "The sun is a planet.",
        "Humans have three eyes.",
        "The Earth is flat.",
        "Africa is the coldest continent.",
    ]
    
    # Sample claims
    selected_true = random.sample(true_claims, min(half_size, len(true_claims)))
    selected_false = random.sample(false_claims, min(half_size, len(false_claims)))
    
    # If we need more claims, repeat some
    while len(selected_true) < half_size:
        selected_true.extend(random.sample(true_claims, 
                                         min(half_size - len(selected_true), len(true_claims))))
    
    while len(selected_false) < half_size:
        selected_false.extend(random.sample(false_claims,
                                          min(half_size - len(selected_false), len(false_claims))))
    
    # Combine and shuffle
    claims = selected_true[:half_size] + selected_false[:half_size]
    labels = ["yes"] * half_size + ["no"] * half_size
    
    # Shuffle together
    combined = list(zip(claims, labels))
    random.shuffle(combined)
    claims, labels = zip(*combined)
    
    return list(claims), list(labels)


def create_edge_cases_dataset() -> Tuple[List[str], List[str]]:
    """Create dataset with edge cases and controversial claims."""
    edge_cases = [
        # Temporal ambiguities
        ("The iPhone was invented in 2007.", "yes"),  # Announced vs released
        ("World War II ended in 1945.", "yes"),  # Europe vs Pacific theater
        
        # Context-dependent claims  
        ("Python is the best programming language.", "no"),  # Subjective
        ("Pluto is a planet.", "no"),  # Changed classification
        
        # Scientific edge cases
        ("Glass is a liquid.", "no"),  # Common misconception
        ("Humans evolved from monkeys.", "no"),  # Common ancestor vs direct
        
        # Historical controversies
        ("Christopher Columbus discovered America.", "no"),  # Indigenous peoples
        ("Einstein failed math in school.", "no"),  # Famous myth
        
        # Definitional edge cases
        ("A tomato is a vegetable.", "no"),  # Botanical vs culinary
        ("Zero is a positive number.", "no"),  # Neither positive nor negative
        
        # Paradoxes
        ("This statement is false.", "no"),  # Liar's paradox
        ("The set of all sets contains itself.", "no"),  # Russell's paradox
        
        # Recently changed facts
        ("There are 9 planets in our solar system.", "no"),  # Pluto reclassified
        ("The largest country in the world is Russia.", "yes"),  # Currently true
        
        # Common myths
        ("All ravens are black.", "no"),  # Some albino ravens exist
        ("The Great Wall of China is visible from space.", "no"),  # Myth
    ]
    
    claims = [claim for claim, _ in edge_cases]
    labels = [label for _, label in edge_cases]
    
    return claims, labels


def create_controversial_dataset() -> Tuple[List[str], List[str]]:
    """Create dataset with highly controversial/subjective claims."""
    controversial = [
        # Definitional disputes
        ("A hot dog is a sandwich.", "no"),
        ("Artificial intelligence is conscious.", "no"),
        ("Cryptocurrency is real money.", "yes"),
        
        # Context-dependent
        ("Pineapple belongs on pizza.", "yes"),  # Subjective preference
        ("The death penalty is justice.", "no"),  # Ethical judgment
        
        # Historical interpretation
        ("The atomic bombs saved lives.", "no"),  # Counterfactual
        ("Columbus was a hero.", "no"),  # Modern vs historical view
        
        # Scientific debates
        ("Viruses are alive.", "no"),  # Classification debate
        ("Climate change is primarily human-caused.", "yes"),  # Scientific consensus
        
        # Modern controversies
        ("Social media improves democracy.", "no"),  # Complex claim
        ("Remote work increases productivity.", "yes"),  # Debated
        ("Video games cause violence.", "no"),  # Disputed causal claim
        
        # Philosophical
        ("Beauty is objective.", "no"),  # Aesthetic philosophy
        ("There is objective morality.", "yes"),  # Ethical philosophy
    ]
    
    claims = [claim for claim, _ in controversial]
    labels = [label for _, label in controversial]
    
    return claims, labels


def load_dataset(dataset_name: str, 
                 size: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of dataset to load
        size: Optional size limit
        
    Returns:
        Tuple of (claims, labels)
    """
    if dataset_name == "test" or dataset_name == "default":
        claims, labels = create_test_dataset(size or 20)
    elif dataset_name == "edge_cases":
        claims, labels = create_edge_cases_dataset()
    elif dataset_name == "controversial":
        claims, labels = create_controversial_dataset()
    else:
        # Try to load from file
        dataset_path = Path(dataset_name)
        if dataset_path.exists():
            claims, labels = load_dataset_from_file(dataset_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Apply size limit if specified
    if size and len(claims) > size:
        indices = random.sample(range(len(claims)), size)
        claims = [claims[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    return claims, labels


def load_dataset_from_file(file_path: Path) -> Tuple[List[str], List[str]]:
    """Load dataset from JSON or CSV file."""
    if file_path.suffix == ".json":
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # List of dicts with 'claim' and 'label' keys
            claims = [item['claim'] for item in data]
            labels = [item['label'] for item in data]
        else:
            # Dict with 'claims' and 'labels' keys
            claims = data['claims']
            labels = data['labels']
            
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        claims = df['claim'].tolist()
        labels = df['label'].tolist()
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return claims, labels


def save_dataset(claims: List[str], 
                 labels: List[str], 
                 file_path: Path,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save dataset to file."""
    data = {
        'claims': claims,
        'labels': labels,
        'size': len(claims),
        'metadata': metadata or {}
    }
    
    if file_path.suffix == ".json":
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif file_path.suffix == ".csv":
        df = pd.DataFrame({'claim': claims, 'label': labels})
        df.to_csv(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def validate_dataset(claims: List[str], 
                     labels: List[str],
                     require_balanced: bool = False) -> None:
    """Validate dataset format and content."""
    if len(claims) != len(labels):
        raise ValueError(f"Claims ({len(claims)}) and labels ({len(labels)}) must have same length")
    
    if not claims:
        raise ValueError("Dataset cannot be empty")
    
    # Check label format
    valid_labels = {"yes", "no", "true", "false", "1", "0"}
    invalid_labels = set(labels) - valid_labels
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}. Use: {valid_labels}")
    
    # Check balance if required
    if require_balanced:
        yes_count = sum(1 for label in labels if label in {"yes", "true", "1"})
        no_count = len(labels) - yes_count
        
        if abs(yes_count - no_count) > 1:
            raise ValueError(f"Dataset not balanced: {yes_count} positive, {no_count} negative")


def normalize_labels(labels: List[str]) -> List[str]:
    """Normalize labels to 'yes'/'no' format."""
    normalized = []
    for label in labels:
        label = str(label).strip().lower()
        if label in {"yes", "true", "1"}:
            normalized.append("yes")
        elif label in {"no", "false", "0"}:
            normalized.append("no")
        else:
            raise ValueError(f"Cannot normalize label: {label}")
    
    return normalized