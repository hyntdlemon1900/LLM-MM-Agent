"""
Quick test script for LLMINA solver
Tests the generated solver without running the full pipeline
"""
import sys
import os
import argparse
import importlib.util

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import FixedINAEvaluation


def load_solver_from_file(solver_path):
    """Dynamically load solver from Python file"""
    spec = importlib.util.spec_from_file_location("llm_solver_module", solver_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.llm_solver


def test_solver(solver_path, topo_name='FatTree', ina_num_list=[3], 
                jobs_num_list=[10], instances_num=10):
    """
    Test a solver implementation
    
    Args:
        solver_path: Path to Python file containing llm_solver function
        topo_name: Network topology ('FatTree' or 'SpineLeaf')
        ina_num_list: List of INA counts to test
        jobs_num_list: List of job counts to test
        instances_num: Number of instances per configuration
    """
    print(f"Loading solver from: {solver_path}")
    solver_func = load_solver_from_file(solver_path)
    
    # Create evaluation class that uses the loaded solver
    class LLMINAEvaluation(FixedINAEvaluation):
        def solver(self, instance):
            """Use the loaded solver function"""
            return solver_func(
                instance=instance,
                network=self.network,
                ina_num=self.ina_num,
                jobs_num=self.jobs_num,
                Cs=self.Cs,
                topo_name=self.topo_name
            )
    
    # Run evaluation
    print(f"\nTesting configuration:")
    print(f"  Topology: {topo_name}")
    print(f"  INA numbers: {ina_num_list}")
    print(f"  Job numbers: {jobs_num_list}")
    print(f"  Instances: {instances_num}")
    print()
    
    evaluator = LLMINAEvaluation(
        topo_name=topo_name,
        ina_num_list=ina_num_list,
        jobs_num_list=jobs_num_list,
        instances_num=instances_num
    )
    
    results = evaluator.evaluate_program()
    
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    for key, value in results.items():
        print(f"{key}: {value}")
    print("=" * 80)
    
    return results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Test LLMINA solver')
    parser.add_argument('solver_path', type=str,
                        help='Path to Python file containing llm_solver function')
    parser.add_argument('--topo_name', type=str, default='FatTree',
                        choices=['FatTree', 'SpineLeaf'],
                        help='Network topology name')
    parser.add_argument('--ina_num', type=int, nargs='+', default=[3],
                        help='List of INA numbers to test')
    parser.add_argument('--jobs_num', type=int, nargs='+', default=[10],
                        help='List of job numbers to test')
    parser.add_argument('--instances_num', type=int, default=10,
                        help='Number of instances for evaluation')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if not os.path.exists(args.solver_path):
        print(f"Error: Solver file not found: {args.solver_path}")
        sys.exit(1)
    
    results = test_solver(
        solver_path=args.solver_path,
        topo_name=args.topo_name,
        ina_num_list=args.ina_num,
        jobs_num_list=args.jobs_num,
        instances_num=args.instances_num
    )
