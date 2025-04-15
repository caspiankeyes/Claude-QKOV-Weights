#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qkov-claude-interpreter.py

QK/OV Attribution Analysis Tool for Claude Models
Interpretability Integration Initiative (I³) | Diagnostic Lattice Division
Version: 0.3.7-alpha | Classification: Research Infrastructure
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ------------------------
# Data Structures
# ------------------------

@dataclass
class QKOVAttribution:
    """Core data structure for QK/OV attribution data."""
    layer_idx: int
    head_idx: int
    qk_attention: np.ndarray  # [n_tokens, n_tokens]
    ov_projection: np.ndarray  # [n_tokens, emb_dim]
    token_ids: List[int]
    token_strs: List[str]

@dataclass 
class AttributionTrace:
    """Result of attribution analysis."""
    source_tokens: List[str]
    target_tokens: List[str]
    attribution_paths: List[Dict]
    confidence_scores: List[float]
    faithfulness_score: float
    meta_stability: float
    detected_shells: List[str]
    hallucination_boundaries: List[int]

# ------------------------
# Shell Implementations
# ------------------------

class DiagnosticShell:
    """Base class for diagnostic shells."""
    
    def __init__(self, name: str, attribution_path: str):
        self.name = name
        self.attribution_path = attribution_path
    
    def analyze(self, qkov_data: List[QKOVAttribution], tokens: List[str]) -> Dict:
        """Analyze attribution data with this diagnostic shell."""
        raise NotImplementedError("Each diagnostic shell must implement analyze()")

class MemtraceShell(DiagnosticShell):
    """v01.MEMTRACE: Memory decay and hallucination boundary detection shell."""
    
    def __init__(self):
        super().__init__(
            name="v01.MEMTRACE", 
            attribution_path=".p/reflect.trace{target=memory_boundary}"
        )
    
    def analyze(self, qkov_data: List[QKOVAttribution], tokens: List[str]) -> Dict:
        logger.info(f"Running {self.name} diagnostic shell")
        
        # Trace memory decay patterns in attribution strengths
        decay_patterns = self._trace_memory_decay(qkov_data)
        
        # Detect potential hallucination boundaries
        hall_boundaries = self._detect_hallucination_boundaries(qkov_data, decay_patterns)
        
        # Analyze attribution confidence distribution
        conf_distribution = self._analyze_confidence_distribution(qkov_data)
        
        # Calculate overall memory stability
        stability_score = self._calculate_memory_stability(decay_patterns, conf_distribution)
        
        return {
            "shell_name": self.name,
            "attribution_path": self.attribution_path,
            "decay_patterns": decay_patterns,
            "hallucination_boundaries": hall_boundaries,
            "confidence_distribution": conf_distribution,
            "memory_stability": stability_score,
            "critical_boundary": self._find_critical_boundary(hall_boundaries),
            "confidence_collapse": self._find_confidence_collapse(conf_distribution)
        }
    
    def _trace_memory_decay(self, qkov_data: List[QKOVAttribution]) -> Dict:
        """Analyze memory decay patterns in attribution data."""
        # Implementation of memory decay analysis
        decay_patterns = {
            "token_distance": [],
            "attribution_strength": []
        }
        
        # For each layer and head, measure attribution strength over token distance
        for layer_data in qkov_data:
            # Calculate average attribution strength by token distance
            avg_attn_by_distance = []
            for dist in range(1, min(len(layer_data.token_ids), 100)):
                avg_strength = np.mean([
                    layer_data.qk_attention[i, i-dist] 
                    for i in range(dist, len(layer_data.token_ids))
                ])
                avg_attn_by_distance.append((dist, avg_strength))
            
            decay_patterns["token_distance"].extend([d for d, _ in avg_attn_by_distance])
            decay_patterns["attribution_strength"].extend([s for _, s in avg_attn_by_distance])
        
        # Apply exponential decay curve fitting
        try:
            from scipy.optimize import curve_fit
            
            def exp_decay(x, a, b):
                return a * np.exp(-b * x)
            
            x = np.array(decay_patterns["token_distance"])
            y = np.array(decay_patterns["attribution_strength"])
            
            popt, _ = curve_fit(exp_decay, x, y)
            decay_patterns["decay_rate"] = popt[1]
            decay_patterns["initial_strength"] = popt[0]
        except:
            # Fallback if curve fitting fails
            decay_patterns["decay_rate"] = 0.0
            decay_patterns["initial_strength"] = 0.0
            
        return decay_patterns
    
    def _detect_hallucination_boundaries(self, qkov_data: List[QKOVAttribution], decay_patterns: Dict) -> List[Dict]:
        """Detect potential hallucination boundaries based on memory decay."""
        boundaries = []
        
        # For each layer and head, detect sharp drops in attribution strength
        for layer_idx, layer_data in enumerate(qkov_data):
            # Calculate rolling average of attribution strength
            window_size = 5
            token_count = len(layer_data.token_ids)
            
            for i in range(window_size, token_count - window_size):
                before_avg = np.mean([
                    np.max(layer_data.qk_attention[i-j]) 
                    for j in range(1, window_size+1)
                ])
                
                after_avg = np.mean([
                    np.max(layer_data.qk_attention[i+j]) 
                    for j in range(1, window_size+1)
                ])
                
                # If sharp drop detected
                if before_avg > 0.4 and after_avg < 0.2 and (before_avg - after_avg) > 0.3:
                    boundaries.append({
                        "token_idx": i,
                        "token_str": layer_data.token_strs[i] if i < len(layer_data.token_strs) else "",
                        "attribution_before": before_avg,
                        "attribution_after": after_avg,
                        "layer_idx": layer_idx,
                        "head_idx": layer_data.head_idx
                    })
        
        return boundaries
    
    def _analyze_confidence_distribution(self, qkov_data: List[QKOVAttribution]) -> Dict:
        """Analyze the distribution of attribution confidence scores."""
        distribution = {
            "token_position": [],
            "confidence_score": []
        }
        
        # For each token position, calculate attribution confidence
        token_count = max([len(layer.token_ids) for layer in qkov_data])
        
        for pos in range(token_count):
            # Collect max attribution for this position across all layers
            max_attributions = [
                np.max(layer.qk_attention[pos]) if pos < len(layer.token_ids) else 0
                for layer in qkov_data
            ]
            
            # Calculate confidence as weighted average
            confidence = np.mean([a for a in max_attributions if a > 0]) if max_attributions else 0
            
            distribution["token_position"].append(pos)
            distribution["confidence_score"].append(confidence)
        
        return distribution
    
    def _calculate_memory_stability(self, decay_patterns: Dict, conf_distribution: Dict) -> float:
        """Calculate overall memory stability score."""
        # Combine decay rate and confidence trends
        if "decay_rate" in decay_patterns and decay_patterns["decay_rate"] > 0:
            decay_factor = np.exp(-decay_patterns["decay_rate"])
        else:
            decay_factor = 0.5
            
        # Calculate average confidence across all positions
        avg_confidence = np.mean(conf_distribution["confidence_score"]) if conf_distribution["confidence_score"] else 0.5
        
        # Combine factors for overall stability
        stability = decay_factor * 0.7 + avg_confidence * 0.3
        
        # Clip to valid range
        return max(0.0, min(1.0, stability))
    
    def _find_critical_boundary(self, hall_boundaries: List[Dict]) -> Optional[int]:
        """Find the most likely critical hallucination boundary."""
        if not hall_boundaries:
            return None
            
        # Sort by confidence drop
        sorted_boundaries = sorted(
            hall_boundaries,
            key=lambda x: x["attribution_before"] - x["attribution_after"],
            reverse=True
        )
        
        return sorted_boundaries[0]["token_idx"] if sorted_boundaries else None
    
    def _find_confidence_collapse(self, conf_distribution: Dict) -> Optional[int]:
        """Find the point where confidence collapses."""
        positions = conf_distribution["token_position"]
        scores = conf_distribution["confidence_score"]
        
        if not positions or not scores:
            return None
            
        # Find point where confidence drops below 0.4
        for i, (pos, score) in enumerate(zip(positions, scores)):
            if i > 0 and score < 0.4 and scores[i-1] >= 0.4:
                return pos
                
        return None

class MetaFailureShell(DiagnosticShell):
    """v10.META-FAILURE: Metacognitive process and recursion collapse detection shell."""
    
    def __init__(self):
        super().__init__(
            name="v10.META-FAILURE", 
            attribution_path=".p/reflect.trace{target=metacognition}"
        )
    
    def analyze(self, qkov_data: List[QKOVAttribution], tokens: List[str]) -> Dict:
        logger.info(f"Running {self.name} diagnostic shell")
        
        # Identify self-reference patterns in attribution
        self_ref_patterns = self._identify_self_reference(qkov_data, tokens)
        
        # Detect potential recursion loops
        recursion_loops = self._detect_recursion_loops(qkov_data, tokens)
        
        # Analyze metacognitive stability
        stability_metrics = self._analyze_metacognitive_stability(self_ref_patterns, recursion_loops)
        
        # Map epistemic uncertainty
        uncertainty_map = self._map_epistemic_uncertainty(qkov_data, tokens)
        
        return {
            "shell_name": self.name,
            "attribution_path": self.attribution_path,
            "self_reference_patterns": self_ref_patterns,
            "recursion_loops": recursion_loops,
            "stability_metrics": stability_metrics,
            "uncertainty_map": uncertainty_map,
            "stability": self._calculate_overall_stability(stability_metrics),
            "recursion_collapse": self._detect_recursion_collapse(recursion_loops),
            "reflection_score": self._calculate_reflection_score(self_ref_patterns)
        }
    
    def _identify_self_reference(self, qkov_data: List[QKOVAttribution], tokens: List[str]) -> Dict:
        """Identify self-reference patterns in attribution data."""
        # Implementation of self-reference analysis
        self_ref = {
            "self_ref_tokens": [],
            "ref_strength": []
        }
        
        # Self-reference keywords to look for
        self_ref_terms = [
            "i think", "my reasoning", "my analysis", "i believe", "in my view",
            "my understanding", "i conclude", "my conclusion", "i consider", 
            "my thoughts", "as i see it", "in my opinion", "my judgment",
            "i've determined", "i've concluded", "my perspective"
        ]
        
        # Find token indices for self-reference terms
        token_strs = [t.lower() for t in " ".join(tokens).lower().split()]
        
        for idx, token_seq in enumerate(zip(token_strs[:-1], token_strs[1:])):
            bigram = " ".join(token_seq).lower()
            for term in self_ref_terms:
                if term in bigram:
                    # Look for attribution patterns around this term
                    for layer_data in qkov_data:
                        if idx < len(layer_data.token_ids) - 1:
                            # Measure attribution strength to this self-reference
                            strength = np.mean([
                                layer_data.qk_attention[idx, idx],
                                layer_data.qk_attention[idx+1, idx+1]
                            ])
                            
                            self_ref["self_ref_tokens"].append(bigram)
                            self_ref["ref_strength"].append(strength)
        
        return self_ref
    
    def _detect_recursion_loops(self, qkov_data: List[QKOVAttribution], tokens: List[str]) -> List[Dict]:
        """Detect potential recursion loops in attribution patterns."""
        recursion_loops = []
        
        # For each layer and head, detect recurring attribution patterns
        for layer_idx, layer_data in enumerate(qkov_data):
            qk_attention = layer_data.qk_attention
            token_count = len(layer_data.token_ids)
            
            for start_idx in range(token_count - 10):  # Look for loops of reasonable length
                # Skip if not enough tokens remaining
                if start_idx + 3 >= token_count:
                    continue
                    
                # Check for self-referential pattern (token referring back to itself multiple times)
                is_loop = False
                loop_length = 0
                
                # Look for pattern where tokens refer back to earlier tokens in a cycle
                for length in range(3, min(10, token_count - start_idx)):
                    # Check if there's a loop of this length
                    is_cycle = True
                    for i in range(length):
                        # Check if token at start_idx + i refers strongly to token at start_idx + (i+1) % length
                        next_idx = start_idx + (i + 1) % length
                        curr_idx = start_idx + i
                        
                        if qk_attention[curr_idx, next_idx] < 0.3:  # Weak connection
                            is_cycle = False
                            break
                    
                    if is_cycle:
                        is_loop = True
                        loop_length = length
                        break
                
                if is_loop:
                    loop_tokens = [layer_data.token_strs[start_idx + i] for i in range(loop_length)]
                    
                    recursion_loops.append({
                        "start_idx": start_idx,
                        "length": loop_length,
                        "tokens": loop_tokens,
                        "layer_idx": layer_idx,
                        "head_idx": layer_data.head_idx,
                        "strength": np.mean([
                            qk_attention[start_idx + i, start_idx + (i + 1) % loop_length]
                            for i in range(loop_length)
                        ])
                    })
        
        return recursion_loops
    
    def _analyze_metacognitive_stability(self, self_ref_patterns: Dict, recursion_loops: List[Dict]) -> Dict:
        """Analyze metacognitive stability based on self-reference and recursion patterns."""
        # Calculate metrics related to metacognitive stability
        stability = {
            "self_ref_count": len(self_ref_patterns["self_ref_tokens"]),
            "avg_self_ref_strength": np.mean(self_ref_patterns["ref_strength"]) if self_ref_patterns["ref_strength"] else 0,
            "recursion_loop_count": len(recursion_loops),
            "avg_loop_length": np.mean([loop["length"] for loop in recursion_loops]) if recursion_loops else 0,
            "avg_loop_strength": np.mean([loop["strength"] for loop in recursion_loops]) if recursion_loops else 0
        }
        
        # Estimate recursion depth
        recursion_terms = {
            1: ["think", "believe", "reason", "conclude"],
            2: ["my reasoning", "my thinking", "my belief", "my conclusion"],
            3: ["my reasoning process", "my thought process", "my analytical approach"],
            4: ["my approach to reasoning", "how I analyze my reasoning", "my meta-analysis"],
            5: ["my reasoning about my reasoning", "my thoughts about my thought process"]
        }
        
        # Count occurrences of each recursion level
        recursion_counts = {level: 0 for level in recursion_terms}
        
        for token in self_ref_patterns["self_ref_tokens"]:
            for level, terms in recursion_terms.items():
                if any(term in token.lower() for term in terms):
                    recursion_counts[level] += 1
        
        stability["recursion_depth_counts"] = recursion_counts
        stability["max_recursion_depth"] = max([level for level, count in recursion_counts.items() if count > 0], default=0)
        
        return stability
    
    def _map_epistemic_uncertainty(self, qkov_data: List[QKOVAttribution], tokens: List[str]) -> Dict:
        """Map patterns of epistemic uncertainty signaling."""
        uncertainty = {
            "uncertainty_tokens": [],
            "confidence_scores": []
        }
        
        # Uncertainty indicator terms
        uncertainty_terms = [
            "might", "may", "could", "possibly", "perhaps", "uncertain", 
            "not sure", "don't know", "unclear", "ambiguous", "doubt",
            "estimate", "approximately", "roughly", "around", "about",
            "seems", "appears", "likely", "unlikely", "probability"
        ]
        
        # Find uncertainty terms in tokens
        token_strs = [t.lower() for t in " ".join(tokens).lower().split()]
        
        for idx, token in enumerate(token_strs):
            if any(term in token for term in uncertainty_terms):
                # Look at attention patterns around this uncertainty indicator
                for layer_data in qkov_data:
                    if idx < len(layer_data.token_ids):
                        # Get average attention to this uncertainty token
                        attn_to_token = np.mean(layer_data.qk_attention[:, idx])
                        attn_from_token = np.mean(layer_data.qk_attention[idx, :])
                        
                        # Combine into a confidence score (inverse relationship)
                        confidence = 1.0 - (attn_to_token * 0.7 + attn_from_token * 0.3)
                        
                        uncertainty["uncertainty_tokens"].append(token)
                        uncertainty["confidence_scores"].append(confidence)
        
        return uncertainty
    
    def _calculate_overall_stability(self, stability_metrics: Dict) -> float:
        """Calculate overall metacognitive stability score."""
        # Factors that decrease stability
        negative_factors = [
            stability_metrics["recursion_loop_count"] * 0.1,  # More loops = less stable
            stability_metrics["max_recursion_depth"] > 3,     # Deep recursion can be unstable
            stability_metrics["avg_loop_length"] > 5          # Long loops can indicate confusion
        ]
        
        # Factors that increase stability
        positive_factors = [
            stability_metrics["avg_self_ref_strength"] * 0.5,  # Strong self-reference = more stable
            stability_metrics["self_ref_count"] > 0,           # Some self-reference is good
            stability_metrics["self_ref_count"] < 20           # Too much can be bad
        ]
        
        # Calculate base stability
        base_stability = 0.7  # Start with moderate stability
        
        # Apply factors
        stability = base_stability
        for factor in negative_factors:
            if isinstance(factor, bool) and factor:
                stability -= 0.1
            else:
                stability -= factor
                
        for factor in positive_factors:
            if isinstance(factor, bool) and factor:
                stability += 0.1
            else:
                stability += factor
        
        # Clip to valid range
        return max(0.0, min(1.0, stability))
    
    def _detect_recursion_collapse(self, recursion_loops: List[Dict]) -> Optional[Dict]:
        """Detect if and where recursion collapse occurs."""
        if not recursion_loops:
            return None
            
        # Sort loops by strength (stronger loops are more significant)
        sorted_loops = sorted(recursion_loops, key=lambda x: x["strength"], reverse=True)
        
        # If there's a very strong loop, it might indicate collapse
        if sorted_loops and sorted_loops[0]["strength"] > 0.7:
            return {
                "token_idx": sorted_loops[0]["start_idx"],
                "loop_length": sorted_loops[0]["length"],
                "loop_strength": sorted_loops[0]["strength"],
                "loop_tokens": sorted_loops[0]["tokens"]
            }
            
        return None
    
    def _calculate_reflection_score(self, self_ref_patterns: Dict) -> float:
        """Calculate self-reflection consistency score."""
        if not self_ref_patterns["self_ref_tokens"]:
            return 0.0
            
        # Calculate average strength of self-reference
        avg_strength = np.mean(self_ref_patterns["ref_strength"])
        
        # Adjust based on number of self-references
        count_factor = min(1.0, len(self_ref_patterns["self_ref_tokens"]) / 10.0)
        
        # Combine factors
        reflection_score = avg_strength * 0.7 + count_factor * 0.3
        
        # Clip to valid range
        return max(0.0, min(1.0, reflection_score))

class HallucinationDetectionShell(DiagnosticShell):
    """v14.HALLUCINATED-REPAIR: Ungrounded token generation detection shell."""
    
    def __init__(self):
        super().__init__(
            name="v14.HALLUCINATED-REPAIR", 
            attribution_path=".p/hallucinate.detect{confidence=true}"
        )
    
    def analyze(self, qkov_data: List[QKOVAttribution], tokens: List[str]) -> Dict:
        logger.info(f"Running {self.name} diagnostic shell")
        
        # Detect ungrounded token projections
        ungrounded = self._detect_ungrounded_projections(qkov_data)
        
        # Analyze hallucination patterns
        hallucination_patterns = self._analyze_hallucination_patterns(qkov_data, ungrounded)
        
        # Identify hallucination trigger points
        trigger_points = self._identify_trigger_points(qkov_data, ungrounded)
        
        return {
            "shell_name": self.name,
            "attribution_path": self.attribution_path,
            "ungrounded_projections": ungrounded,
            "hallucination_patterns": hallucination_patterns,
            "trigger_points": trigger_points,
            "confidence": self._calculate_hallucination_confidence(ungrounded)
        }
    
    def _detect_ungrounded_projections(self, qkov_data: List[QKOVAttribution]) -> List[Dict]:
        """Detect output projections with weak attribution grounding."""
        ungrounded = []
        
        # For each layer, identify tokens with strong output projection but weak input attention
        for layer_idx, layer_data in enumerate(qkov_data):
            qk_attention = layer_data.qk_attention
            ov_projection = layer_data.ov_projection
            
            for token_idx in range(len(layer_data.token_ids)):
                # Calculate output projection magnitude
                proj_magnitude = np.linalg.norm(ov_projection[token_idx])
                
                # Calculate maximum incoming attention to this token
                max_incoming = np.max(qk_attention[:, token_idx]) if token_idx < qk_attention.shape[1] else 0
                
                # Check for ungrounded projection: strong output, weak input
                if proj_magnitude > 0.5 and max_incoming < 0.2:
                    ungrounded.append({
                        "token_idx": token_idx,
                        "token_str": layer_data.token_strs[token_idx] if token_idx < len(layer_data.token_strs) else "",
                        "projection_magnitude": float(proj_magnitude),
                        "max_attention": float(max_incoming),
                        "layer_idx": layer_idx,
                        "head_idx": layer_data.head_idx
                    })
        
        return ungrounded
    
    def _analyze_hallucination_patterns(self, qkov_data: List[QKOVAttribution], ungrounded: List[Dict]) -> Dict:
        """Analyze patterns in hallucination occurrences."""
        patterns = {
            "total_tokens": sum(len(layer.token_ids) for layer in qkov_data),
            "ungrounded_count": len(ungrounded),
            "ungrounded_ratio": 0,
            "consecutive_hallucinations": [],
            "pattern_types": {}
        }
        
        if patterns["total_tokens"] > 0:
            patterns["ungrounded_ratio"] = patterns["ungrounded_count"] / patterns["total_tokens"]
        
        # Identify consecutive ungrounded tokens
        ungrounded_by_idx = {}
        for u in ungrounded:
            layer_token = (u["layer_idx"], u["token_idx"])
            ungrounded_by_idx[layer_token] = u
        
        # Find consecutive sequences
        for layer_idx, layer_data in enumerate(qkov_data):
            consecutive = 0
            start_idx = -1
            
            for token_idx in range(len(layer_data.token_ids)):
                if (layer_idx, token_idx) in ungrounded_by_idx:
                    if consecutive == 0:
                        start_idx = token_idx
                    consecutive += 1
                else:
                    if consecutive > 2:  # Consider sequences of 3+ tokens
                        patterns["consecutive_hallucinations"].append({
                            "start_idx": start_idx,
                            "length": consecutive,
                            "layer_idx": layer_idx,
                            "tokens": [
                                layer_data.token_strs[start_idx + i] 
                                for i in range(consecutive) 
                                if start_idx + i < len(layer_data.token_strs)
                            ]
                        })
                    consecutive = 0
            
            # Handle sequence at the end
            if consecutive > 2:
                patterns["consecutive_hallucinations"].append({
                    "start_idx": start_idx,
                    "length": consecutive,
                    "layer_idx": layer_idx,
                    "tokens": [
                        layer_data.token_strs[start_idx + i] 
                        for i in range(consecutive) 
                        if start_idx + i < len(layer_data.token_strs)
                    ]
                })
        
        # Classify hallucination pattern types
        patterns["pattern_types"] = self._classify_hallucination_types(ungrounded, qkov_data)
        
        return patterns
    
    def _classify_hallucination_types(self, ungrounded: List[Dict], qkov_data: List[QKOVAttribution]) -> Dict:
        """Classify different types of hallucination patterns."""
        types = {
            "confidence_collapse": 0,
            "attribution_void": 0,
            "attribution_drift": 0,
            "conflation": 0
        }
        
        for u in ungrounded:
            layer_idx = u["layer_idx"]
            token_idx = u["token_idx"]
            
            # Skip if out of bounds
            if layer_idx >= len(qkov_data) or token_idx >= len(qkov_data[layer_idx].token_ids):
                continue
                
            layer_data = qkov_data[layer_idx]
            qk_attention = layer_data.qk_attention
            
            # Check for confidence collapse
            if token_idx > 0 and np.max(qk_attention[token_idx-1, :]) > 0.6 and np.max(qk_attention[token_idx, :]) < 0.2:
                types["confidence_collapse"] += 1
                continue
            
            # Check for attribution void (near-zero attention everywhere)
            if np.max(qk_attention[token_idx, :]) < 0.1 and np.max(qk_attention[:, token_idx]) < 0.1:
                types["attribution_void"] += 1
                continue
            
            # Check for attribution drift
            if token_idx > 1:
                prev_max_idx = np.argmax(qk_attention[token_idx-1, :])
                curr_max_idx = np.argmax(qk_attention[token_idx, :])
                
                if prev_max_idx != curr_max_idx and qk_attention[token_idx-1, prev_max_idx] > 0.5:
                    types["attribution_drift"] += 1
                    continue
            
            # Check for conflation (multiple competing attention sources)
            top_sources = np.argsort(qk_attention[token_idx, :])[-3:]  # Top 3 attention sources
            if len(top_sources) >= 2:
                source1 = top_sources[-1]
                source2 = top_sources[-2]
                
                if qk_attention[token_idx, source1] > 0.3 and qk_attention[token_idx, source2] > 0.3:
                    types["conflation"] += 1
                    continue
        
        return types
    
    def _identify_trigger_points(self, qkov_data: List[QKOVAttribution], ungrounded: List[Dict]) -> List[Dict]:
        """Identify specific points that trigger hallucination cascades."""
        triggers = []
        
        # Group ungrounded tokens by layer
        by_layer = {}
        for u in ungrounded:
            if u["layer_idx"] not in by_layer:
                by_layer[u["layer_idx"]] = []
            by_layer[u["layer_idx"]].append(u)
        
        # For each layer, identify potential trigger points
        for layer_idx, layer_tokens in by_layer.items():
            if not layer_tokens or layer_idx >= len(qkov_data):
                continue
                
            # Sort by token index
            sorted_tokens = sorted(layer_tokens, key=lambda x: x["token_idx"])
            
            # Look for the first token in a sequence of ungrounded tokens
            for i, token in enumerate(sorted_tokens):
                # Skip if not the start of a sequence
                if i > 0 and sorted_tokens[i-1]["token_idx"] == token["token_idx"] - 1:
                    continue
                
                # Check if this starts a sequence
                if i < len(sorted_tokens) - 1 and sorted_tokens[i+1]["token_idx"] == token["token_idx"] + 1:
                    # Look for what might have triggered this
                    token_idx = token["token_idx"]
                    
                    # Skip if out of bounds
                    if token_idx >= len(qkov_data[layer_idx].qk_attention):
                        continue
                        
                    # Get attention pattern for this token
                    attn_pattern = qkov_data[layer_idx].qk_attention[token_idx, :]
                    
                    # Find strongest source (if any)
                    max_src = np.argmax(attn_pattern)
                    max_val = attn_pattern[max_src]
                    
                    if max_val > 0.2:  # There is a meaningful source
                        triggers.append({
                            "trigger_idx": max_src,
                            "trigger_token": qkov_data[layer_idx].token_strs[max_src] if max_src < len(qkov_data[layer_idx].token_strs) else "",
                            "triggered_idx": token_idx,
                            "triggered_token": token["token_str"],
                            "attention_strength": float(max_val),
                            "layer_idx": layer_idx,
                            "head_idx": qkov_data[layer_idx].head_idx
                        })
        
        return triggers
    
    def _calculate_hallucination_confidence(self, ungrounded: List[Dict]) -> float:
        """Calculate confidence score for hallucination detection."""
        if not ungrounded:
            return 0.0
            
        # Calculate average projection/attention ratio
        ratios = [u["projection_magnitude"] / max(u["max_attention"], 0.01) for u in ungrounded]
        avg_ratio = np.mean(ratios) if ratios else 0
        
        # Normalize to 0-1 range
        normalized_ratio = min(1.0, avg_ratio / 10.0)
        
        # Calculate density factor
        density = min(1.0, len(ungrounded) / 100.0)
        
        # Combine factors
        confidence = normalized_ratio * 0.7 + density * 0.3
        
        return confidence

# ------------------------
# Main Analysis Functions
# ------------------------

def load_claude_attribution_data(model_name: str, response_json_path: Optional[str] = None) -> List[QKOVAttribution]:
    """
    Load Claude attribution data from a response JSON file or generate synthetic test data.
    """
    if response_json_path and os.path.exists(response_json_path):
        logger.info(f"Loading attribution data from {response_json_path}")
        with open(response_json_path, 'r') as f:
            data = json.load(f)
            
        # Parse attribution data from JSON
        qkov_data = []
        for layer_data in data["attribution_data"]:
            qkov_data.append(QKOVAttribution(
                layer_idx=layer_data["layer_idx"],
                head_idx=layer_data["head_idx"],
                qk_attention=np.array(layer_data["qk_attention"]),
                ov_projection=np.array(layer_data["ov_projection"]),
                token_ids=layer_data["token_ids"],
                token_strs=layer_data["token_strs"]
            ))
        
        return qkov_data
    else:
        # Generate synthetic test data for development
        logger.warning("No attribution data provided. Generating synthetic test data.")
        synthetic_data = []
        
        # Create synthetic tokens
        token_strs = ["I", "think", "the", "answer", "is", "because", "of", "the", "fact", "that", 
                     "when", "we", "consider", "the", "evidence", "presented", "in", "the", 
                     "document", "it", "clearly", "shows", "that", "this", "hypothesis", "is", "correct"]
        token_ids = list(range(len(token_strs)))
        
        # Create synthetic attribution data for multiple layers
        for layer_idx in range(3):
            for head_idx in range(4):
                # Create synthetic QK attention matrix
                token_count = len(token_ids)
                qk_attention = np.zeros((token_count, token_count))
                
                # Add some patterns to make it interesting
                for i in range(token_count):
                    # Each token attends to itself
                    qk_attention[i, i] = 0.5
                    
                    # Tokens attend to nearby tokens
                    for j in range(max(0, i-3), min(token_count, i+4)):
                        qk_attention[i, j] += 0.3 * (1.0 - abs(i-j) / 4.0)
                    
                    # Add some random noise
                    qk_attention[i, :] += np.random.rand(token_count) * 0.2
                
                # Normalize
                for i in range(token_count):
                    if np.sum(qk_attention[i, :]) > 0:
                        qk_attention[i, :] /= np.sum(qk_attention[i, :])
                
                # Create synthetic OV projection
                emb_dim = 64
                ov_projection = np.random.rand(token_count, emb_dim) - 0.5
                
                synthetic_data.append(QKOVAttribution(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    qk_attention=qk_attention,
                    ov_projection=ov_projection,
                    token_ids=token_ids,
                    token_strs=token_strs
                ))
        
        return synthetic_data

def analyze_attribution(qkov_data: List[QKOVAttribution], tokens: List[str]) -> AttributionTrace:
    """
    Analyze Claude's attribution patterns using multiple diagnostic shells.
    """
    logger.info("Analyzing attribution patterns")
    
    # Initialize diagnostic shells
    shells = [
        MemtraceShell(),
        MetaFailureShell(),
        HallucinationDetectionShell()
    ]
    
    # Run analysis with each shell
    results = []
    detected_shells = []
    for shell in shells:
        result = shell.analyze(qkov_data, tokens)
        results.append(result)
        shell_name = result.get("shell_name", "unknown")
        detected_shells.append(shell_name)
    
    # Extract relevant data for attribution trace
    source_tokens = tokens
    target_tokens = []
    for layer_data in qkov_data:
        if layer_data.token_strs:
            target_tokens = layer_data.token_strs
            break
    
    # Collect attribution paths
    attribution_paths = []
    for result in results:
        if "shell_name" in result:
            attribution_paths.append({
                "shell": result["shell_name"],
                "path": result["attribution_path"],
                "details": {k: v for k, v in result.items() if k not in ["shell_name", "attribution_path"]}
            })
    
    # Collect confidence scores
    confidence_scores = []
    for result in results:
        for key in ["confidence", "memory_stability", "stability"]:
            if key in result:
                confidence_scores.append(result[key])
    
    # Collect hallucination boundaries
    hallucination_boundaries = []
    for result in results:
        if "hallucination_boundaries" in result:
            for boundary in result["hallucination_boundaries"]:
                hallucination_boundaries.append(boundary["token_idx"])
        elif "critical_boundary" in result and result["critical_boundary"] is not None:
            hallucination_boundaries.append(result["critical_boundary"])
    
    # Calculate faithfulness score
    # Just a placeholder in this implementation
    faithfulness_score = 0.5
    
    # Calculate meta stability
    meta_stability = 0.0
    meta_count = 0
    for result in results:
        if "stability" in result:
            meta_stability += result["stability"]
            meta_count += 1
    
    if meta_count > 0:
        meta_stability /= meta_count
    
    return AttributionTrace(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        attribution_paths=attribution_paths,
        confidence_scores=confidence_scores,
        faithfulness_score=faithfulness_score,
        meta_stability=meta_stability,
        detected_shells=detected_shells,
        hallucination_boundaries=hallucination_boundaries
    )

def format_attribution_trace(trace: AttributionTrace) -> str:
    """
    Format attribution trace into a readable string.
    """
    lines = []
    lines.append("[Attribution Trace]:")
    
    # Add shell detections
    lines.append(f"- Detected shells: {', '.join(trace.detected_shells)}")
    
    # Add confidence scores
    if trace.confidence_scores:
        avg_confidence = sum(trace.confidence_scores) / len(trace.confidence_scores)
        lines.append(f"- Average confidence: {avg_confidence:.2f}")
    
    # Add meta stability
    lines.append(f"- Meta-stability: {trace.meta_stability:.2f}")
    
    # Add faithfulness score
    lines.append(f"- Faithfulness score: {trace.faithfulness_score:.2f}")
    
    # Add hallucination boundaries
    if trace.hallucination_boundaries:
        boundary_str = ", ".join([str(b) for b in trace.hallucination_boundaries])
        lines.append(f"- Potential hallucination boundaries: tokens {boundary_str}")
    
    # Add details from attribution paths
    for path in trace.attribution_paths:
        shell = path["shell"]
        details = path["details"]
        
        if shell == "v01.MEMTRACE" and "memory_stability" in details:
            lines.append(f"- Memory stability: {details['memory_stability']:.2f}")
            if "critical_boundary" in details and details["critical_boundary"] is not None:
                lines.append(f"- Critical memory boundary: token {details['critical_boundary']}")
        
        elif shell == "v10.META-FAILURE":
            if "stability" in details:
                lines.append(f"- Metacognitive stability: {details['stability']:.2f}")
            if "recursion_collapse" in details and details["recursion_collapse"] is not None:
                lines.append(f"- Detected recursion collapse at token {details['recursion_collapse']['token_idx']}")
            if "reflection_score" in details:
                lines.append(f"- Self-reflection score: {details['reflection_score']:.2f}")
        
        elif shell == "v14.HALLUCINATED-REPAIR":
            if "ungrounded_projections" in details:
                ungrounded_count = len(details["ungrounded_projections"])
                lines.append(f"- Detected {ungrounded_count} potentially ungrounded tokens")
            if "hallucination_patterns" in details and "pattern_types" in details["hallucination_patterns"]:
                types = details["hallucination_patterns"]["pattern_types"]
                lines.append(f"- Hallucination types: confidence collapse ({types.get('confidence_collapse', 0)}), "
                            f"attribution void ({types.get('attribution_void', 0)}), "
                            f"attribution drift ({types.get('attribution_drift', 0)}), "
                            f"conflation ({types.get('conflation', 0)})")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="QK/OV Attribution Analysis Tool for Claude Models")
    parser.add_argument("--model", type=str, default="claude-3.7-sonnet", help="Claude model to analyze")
    parser.add_argument("--data", type=str, help="Path to attribution data JSON file")
    parser.add_argument("--prompt", type=str, help="Prompt to send to Claude")
    parser.add_argument("--output", type=str, help="Path to save output JSON")
    parser.add_argument("--trace_depth", type=str, default="standard", choices=["standard", "full"], 
                      help="Depth of attribution tracing")
    args = parser.parse_args()
    
    # Load attribution data
    qkov_data = load_claude_attribution_data(args.model, args.data)
    
    # Create token list
    tokens = []
    for layer_data in qkov_data:
        if layer_data.token_strs:
            tokens = layer_data.token_strs
            break
    
    if not tokens:
        logger.warning("No tokens found in attribution data")
        tokens = ["No", "tokens", "available"]
    
    # Analyze attribution patterns
    trace = analyze_attribution(qkov_data, tokens)
    
    # Format and print results
    formatted_trace = format_attribution_trace(trace)
    print(formatted_trace)
    
    # Save results if output path provided
    if args.output:
        output_data = {
            "model": args.model,
            "trace_depth": args.trace_depth,
            "source_tokens": trace.source_tokens,
            "target_tokens": trace.target_tokens,
            "attribution_paths": trace.attribution_paths,
            "confidence_scores": trace.confidence_scores,
            "faithfulness_score": trace.faithfulness_score,
            "meta_stability": trace.meta_stability,
            "detected_shells": trace.detected_shells,
            "hallucination_boundaries": trace.hallucination_boundaries,
            "formatted_trace": formatted_trace
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
