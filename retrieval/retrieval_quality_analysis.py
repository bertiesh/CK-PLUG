"""
Analyzes how CK-PLUG effectiveness varies with retrieval quality by binning 
examples into quintiles and measuring performance within each bin.
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Tuple
from ck_plug_integration import integrate_ck_plug_with_analyzer

import os

class RetrievalQualityAnalyzer:
    def __init__(
        self,
        model_name: str,
        retriever_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda"
    ):
        """
        Initialize analyzer with model and retriever.
        
        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-3-8b-instruct")
            retriever_name: Sentence transformer model for computing retrieval scores
            device: Device to run on
        """
        self.device = device
        self.model_name = model_name
        
        # Load language model
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # Load retriever for computing retrieval quality scores
        print(f"Loading retriever: {retriever_name}")
        self.retriever = SentenceTransformer(retriever_name)
        self.retriever.to(device)
        
    def compute_retrieval_score(self, query: str, context: str) -> float:
        """
        Compute retrieval quality score as cosine similarity between query and context.
        
        Args:
            query: Question text
            context: Retrieved context text
            
        Returns:
            Cosine similarity score [0, 1]
        """
        query_emb = self.retriever.encode(query, convert_to_tensor=True)
        context_emb = self.retriever.encode(context, convert_to_tensor=True)
        
        # Compute cosine similarity
        score = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0), 
            context_emb.unsqueeze(0)
        ).item()
        
        return score
    
    def generate_answer(
        self, 
        query: str, 
        context: str = None, 
        max_new_tokens: int = 64,
        temperature: float = 0.0
    ) -> str:
        """
        Generate answer using standard greedy decoding (baseline).
        
        Args:
            query: Question text
            context: Optional context to prepend
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)
            
        Returns:
            Generated answer text
        """
        if context:
            prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        else:
            prompt = f"Question: {query}\nAnswer:"
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        answer = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return answer.strip()
    
    def generate_with_ck_plug(
        self,
        query: str,
        context: str,
        alpha: float = 0.5,
        epsilon: float = -1.0,
        max_new_tokens: int = 64
    ) -> str:
        """
        Generate answer using CK-PLUG method.

        NOTE: This method will be overridden by the CK-PLUG integration
        (see ck_plug_integration.integrate_ck_plug_with_analyzer).
        If integration is not applied, this falls back to the baseline.
        """
        # Fallback: baseline behavior
        return self.generate_answer(query, context, max_new_tokens=max_new_tokens)
    
    def normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison."""
        import re
        import string
        
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def evaluate_example(
        self,
        example: Dict,
        method: str = "baseline",
        alpha: float = 0.5
    ) -> Dict:
        """
        Evaluate a single example and return metrics.
        
        Args:
            example: Dictionary with 'question', 'context', 'answer' keys
            method: "baseline", "ck_plug", or "conf_filter"
            alpha: CK-PLUG alpha parameter
            
        Returns:
            Dictionary with results including retrieval_score, prediction, correct
        """
        query = example['question']
        context = example.get('context', '')
        gold_answer = example['answer']
        
        # Compute retrieval quality score
        retrieval_score = self.compute_retrieval_score(query, context)
        
        # Generate prediction based on method
        if method == "baseline":
            prediction = self.generate_answer(query, context)
        elif method == "ck_plug":
            prediction = self.generate_with_ck_plug(query, context, alpha=alpha)
        elif method == "conf_filter":
            # Confidence filtering: only use context if score > threshold
            threshold = 0.6  # Can be tuned
            if retrieval_score >= threshold:
                prediction = self.generate_answer(query, context)
            else:
                prediction = self.generate_answer(query, None)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Check correctness
        pred_norm = self.normalize_answer(prediction)
        gold_norm = self.normalize_answer(gold_answer)
        correct = gold_norm in pred_norm or pred_norm in gold_norm
        
        return {
            'question': query,
            'prediction': prediction,
            'gold_answer': gold_answer,
            'retrieval_score': retrieval_score,
            'correct': correct,
            'method': method
        }
    
    def analyze_dataset(
        self,
        dataset: List[Dict],
        methods: List[str] = ["baseline", "ck_plug", "conf_filter"],
        alpha: float = 0.5,
        n_quintiles: int = 5
    ) -> pd.DataFrame:
        """
        Analyze entire dataset across retrieval quality quintiles.
        
        Args:
            dataset: List of examples
            methods: Methods to evaluate
            alpha: CK-PLUG alpha parameter
            n_quintiles: Number of quality bins
            
        Returns:
            DataFrame with results per quintile and method
        """
        results = []
        
        # First pass: compute retrieval scores for all examples
        print("Computing retrieval scores...")
        for example in tqdm(dataset):
            query = example['question']
            context = example.get('context', '')
            score = self.compute_retrieval_score(query, context)
            example['retrieval_score'] = score
        
        # Determine quintile boundaries
        scores = [ex['retrieval_score'] for ex in dataset]
        quintile_boundaries = np.percentile(scores, np.linspace(0, 100, n_quintiles + 1))
        
        print(f"\nQuintile boundaries: {quintile_boundaries}")
        
        # Assign examples to quintiles
        for example in dataset:
            score = example['retrieval_score']
            quintile = np.digitize(score, quintile_boundaries[1:-1])
            example['quintile'] = quintile
        
        # Evaluate each method
        for method in methods:
            print(f"\nEvaluating method: {method}")
            
            for example in tqdm(dataset):
                result = self.evaluate_example(example, method=method, alpha=alpha)
                result['quintile'] = example['quintile']
                results.append(result)
        
        return pd.DataFrame(results)
    
    def compute_quintile_metrics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute accuracy metrics per quintile and method.
        
        Args:
            results_df: DataFrame from analyze_dataset
            
        Returns:
            DataFrame with quintile-level metrics
        """
        quintile_metrics = []
        
        for method in results_df['method'].unique():
            method_df = results_df[results_df['method'] == method]
            
            for quintile in sorted(method_df['quintile'].unique()):
                quintile_df = method_df[method_df['quintile'] == quintile]
                
                accuracy = quintile_df['correct'].mean() * 100
                avg_score = quintile_df['retrieval_score'].mean()
                count = len(quintile_df)
                
                quintile_metrics.append({
                    'method': method,
                    'quintile': quintile,
                    'accuracy': accuracy,
                    'avg_retrieval_score': avg_score,
                    'count': count
                })
        
        return pd.DataFrame(quintile_metrics)
    
    def plot_retrieval_quality_analysis(
        self,
        quintile_metrics: pd.DataFrame,
        save_path: str = "retrieval_quality_analysis.png"
    ):
        """
        Create visualization of performance across retrieval quality quintiles.
        
        Args:
            quintile_metrics: DataFrame from compute_quintile_metrics
            save_path: Path to save figure
        """
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Accuracy across quintiles
        plt.subplot(1, 2, 1)
        for method in quintile_metrics['method'].unique():
            method_df = quintile_metrics[quintile_metrics['method'] == method]
            plt.plot(
                method_df['quintile'], 
                method_df['accuracy'],
                marker='o',
                label=method,
                linewidth=2
            )
        
        plt.xlabel('Retrieval Quality Quintile', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Performance vs. Retrieval Quality', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(sorted(quintile_metrics['quintile'].unique()))
        
        # Plot 2: Improvement over baseline
        plt.subplot(1, 2, 2)
        baseline_df = quintile_metrics[quintile_metrics['method'] == 'baseline']
        
        for method in quintile_metrics['method'].unique():
            if method == 'baseline':
                continue
            
            method_df = quintile_metrics[quintile_metrics['method'] == method]
            improvements = []
            quintiles = []
            
            for quintile in sorted(method_df['quintile'].unique()):
                baseline_acc = baseline_df[baseline_df['quintile'] == quintile]['accuracy'].values[0]
                method_acc = method_df[method_df['quintile'] == quintile]['accuracy'].values[0]
                improvement = method_acc - baseline_acc
                improvements.append(improvement)
                quintiles.append(quintile)
            
            plt.plot(quintiles, improvements, marker='o', label=method, linewidth=2)
        
        plt.xlabel('Retrieval Quality Quintile', fontsize=12)
        plt.ylabel('Accuracy Improvement over Baseline (%)', fontsize=12)
        plt.title('Improvement vs. Retrieval Quality', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xticks(quintiles)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.close()
    
    def generate_latex_table(
        self,
        quintile_metrics: pd.DataFrame,
        save_path: str = "quintile_metrics_table.tex"
    ):
        """
        Generate LaTeX table of quintile metrics.
        
        Args:
            quintile_metrics: DataFrame from compute_quintile_metrics
            save_path: Path to save LaTeX file
        """
        # Pivot to get methods as columns
        pivot_df = quintile_metrics.pivot_table(
            index='quintile',
            columns='method',
            values='accuracy',
            aggfunc='first'
        ).reset_index()
        
        # Add average retrieval score column
        avg_scores = quintile_metrics.groupby('quintile')['avg_retrieval_score'].first()
        pivot_df['Avg Retrieval Score'] = pivot_df['quintile'].map(avg_scores)
        
        # Reorder columns
        cols = ['quintile', 'Avg Retrieval Score'] + [col for col in pivot_df.columns if col not in ['quintile', 'Avg Retrieval Score']]
        pivot_df = pivot_df[cols]
        
        # Rename quintile column
        pivot_df['quintile'] = 'Q' + pivot_df['quintile'].astype(str)
        pivot_df = pivot_df.rename(columns={'quintile': 'Quintile'})
        
        # Format numbers
        for col in pivot_df.columns:
            if col != 'Quintile':
                if 'Score' in col:
                    pivot_df[col] = pivot_df[col].apply(lambda x: f"{x:.2f}")
                else:
                    pivot_df[col] = pivot_df[col].apply(lambda x: f"{x:.1f}")
        
        # Generate LaTeX
        latex_str = pivot_df.to_latex(
            index=False,
            escape=False,
            column_format='l' + 'c' * (len(pivot_df.columns) - 1)
        )
        
        with open(save_path, 'w') as f:
            f.write(latex_str)
        
        print(f"LaTeX table saved to: {save_path}")


def load_nq_dataset(file_path: str, max_examples: int = None) -> List[Dict]:
    """
    Load Natural Questions dataset.
    
    Args:
        file_path: Path to JSONL file
        max_examples: Maximum number of examples to load
        
    Returns:
        List of example dictionaries
    """
    examples = []
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            
            data = json.loads(line)
            
            # Extract question, context, and answer
            # Adjust keys based on your dataset format
            example = {
                'question': data.get('question', data.get('input', '')),
                'context': data.get('context', data.get('ctxs', [''])[0] if 'ctxs' in data else ''),
                'answer': data.get('answer', data.get('answers', [''])[0] if 'answers' in data else '')
            }
            
            # Handle various answer formats
            if isinstance(example['answer'], list):
                example['answer'] = example['answer'][0] if example['answer'] else ''
            
            examples.append(example)
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Retrieval Quality Dependency Analysis")
    parser.add_argument("--model_name", type=str, required=True,
                       help="HuggingFace model name")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset JSONL file")
    parser.add_argument("--output_dir", type=str, default="./retrieval_analysis_results",
                       help="Directory to save results")
    parser.add_argument("--max_examples", type=int, default=1000,
                       help="Maximum number of examples to evaluate")
    parser.add_argument("--n_quintiles", type=int, default=5,
                       help="Number of retrieval quality bins")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="CK-PLUG alpha parameter")
    parser.add_argument("--methods", nargs='+', default=["baseline", "ck_plug", "conf_filter"],
                       help="Methods to evaluate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    print("Initializing analyzer...")
    analyzer = RetrievalQualityAnalyzer(model_name=args.model_name)

    # Attach CK-PLUG implementation to analyzer
    analyzer = integrate_ck_plug_with_analyzer(analyzer)
    
    # Load dataset
    print(f"\nLoading dataset from: {args.dataset_path}")
    dataset = load_nq_dataset(args.dataset_path, max_examples=args.max_examples)
    print(f"Loaded {len(dataset)} examples")
    
    # Run analysis
    print("\nStarting retrieval quality analysis...")
    results_df = analyzer.analyze_dataset(
        dataset,
        methods=args.methods,
        alpha=args.alpha,
        n_quintiles=args.n_quintiles
    )
    
    # Save raw results
    results_path = os.path.join(args.output_dir, "raw_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nRaw results saved to: {results_path}")
    
    # Compute quintile metrics
    print("\nComputing quintile-level metrics...")
    quintile_metrics = analyzer.compute_quintile_metrics(results_df)
    
    # Save quintile metrics
    metrics_path = os.path.join(args.output_dir, "quintile_metrics.csv")
    quintile_metrics.to_csv(metrics_path, index=False)
    print(f"Quintile metrics saved to: {metrics_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("QUINTILE ANALYSIS SUMMARY")
    print("="*80)
    print(quintile_metrics.to_string(index=False))
    print("="*80)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_path = os.path.join(args.output_dir, "retrieval_quality_plot.png")
    analyzer.plot_retrieval_quality_analysis(quintile_metrics, save_path=plot_path)
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    latex_path = os.path.join(args.output_dir, "quintile_table.tex")
    analyzer.generate_latex_table(quintile_metrics, save_path=latex_path)
    
    # Compute and print key statistics
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    for method in quintile_metrics['method'].unique():
        if method == 'baseline':
            continue
        
        method_df = quintile_metrics[quintile_metrics['method'] == method]
        baseline_df = quintile_metrics[quintile_metrics['method'] == 'baseline']
        
        # Get Q1 (lowest quality) and Q5 (highest quality)
        q1_improvement = (
            method_df[method_df['quintile'] == 1]['accuracy'].values[0] -
            baseline_df[baseline_df['quintile'] == 1]['accuracy'].values[0]
        )
        
        q5_improvement = (
            method_df[method_df['quintile'] == 5]['accuracy'].values[0] -
            baseline_df[baseline_df['quintile'] == 5]['accuracy'].values[0]
        )
        
        print(f"\n{method.upper()}:")
        print(f"  Q1 (lowest quality) improvement: +{q1_improvement:.2f}%")
        print(f"  Q5 (highest quality) improvement: +{q5_improvement:.2f}%")
        print(f"  Improvement range: {q1_improvement - q5_improvement:.2f} percentage points")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()