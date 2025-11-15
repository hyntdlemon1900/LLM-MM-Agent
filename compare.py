
"""
Example usage of PyomoSolver with comparison to heuristic algorithm.

This creates test instances and compares MILP solver with heuristic.
"""
import time
from topo import FatTree
from dataset import generate_dataset_job_random_fattree
from evaluation import FixedINAEvaluation, evaluate_completion_time
from Pyomo import *

print("="*80)
print("LLMINA Solver Comparison: MILP vs Heuristic")
print("="*80)

# Create network topology
print("\n[1] Creating FatTree topology...")
k = 4                    # k-ary FatTree
basic_band = 100         # Base bandwidth (Gbps)
hosts_num_in_fat = 20    # Hosts per ToR switch
k1 = 0.5                 # Link capacity multiplier

network = FatTree(k, basic_band, hosts_num_in_fat, 'fixed', k1)
all_workers_id = network.all_workers_id

print(f"✓ FatTree topology created:")
print(f"  - k = {k}")
print(f"  - Total workers: {len(all_workers_id)}")
print(f"  - ToR switches: {len(network.tors_id)}")
print(f"  - Aggregation switches: {len(network.aggrs_id)}")
print(f"  - Core switches: {len(network.cores_id)}")

# Generate test instances
print("\n[2] Generating problem instances...")
jobs_num = 3
instances_num = 1
K = 5
Cs = 750.0
Ps = 200.0

dataset = generate_dataset_job_random_fattree(
    jobs_num, 
    all_workers_id, 
    instances_num,
    len(network.tors_id),  # num_edge_switches
    True
)

print(f"✓ Generated {instances_num} instances")
print(f"  - Jobs per instance: {jobs_num}")
print(f"  - INA budget K: {K}")

# Initialize results storage
heuristic_results = []
milp_results = []

# Process each instance
for idx, (name, instance) in enumerate(dataset.items(), 1):
    print(f"\n{'='*80}")
    print(f"Instance {idx}/{instances_num}")
    print(f"{'='*80}")
    print(f"Workers per job: {instance['workers_num']}")
    print(f"Job sizes: {[f'{s:.1f}' for s in instance['jobs_size']]}")
    
    # ========== Run Heuristic Algorithm ==========
    print(f"\n[Heuristic Algorithm]")
    evaluator = FixedINAEvaluation(
        topo_name='FatTree',
        ina_num_list=[K],
        jobs_num_list=[jobs_num],
        instances_num=1
    )
    evaluator.network = network
    evaluator.ina_num = K
    evaluator.jobs_num = jobs_num
    
    h_start = time.time()
    h_placement, h_routing = evaluator.solver(instance)
    h_runtime = time.time() - h_start
    
    # Convert numpy arrays to lists for compatibility
    if isinstance(h_routing[1], np.ndarray):
        h_routing = [h_routing[0], h_routing[1].tolist()]
    
    h_makespan = evaluate_completion_time(
        instance=instance,
        network=network,
        ina_placement=h_placement,
        jobs_routing=h_routing,
        verbose=True,  # 改为True以显示错误信息
        ina_capacity=Cs,
        ps_bandwidth=Ps
    )
    
    print(f"  ✓ Makespan: {h_makespan:.4f}")
    print(f"  ✓ Runtime: {h_runtime:.4f}s")
    
    heuristic_results.append({
        'makespan': h_makespan,
        'runtime': h_runtime,
        'placement': h_placement
    })
    
    # ========== Run MILP Solver ==========
    print(f"\n[MILP Solver]")
    print("-"*80)
    
    m_start = time.time()
    try:
        milp_solver = PyomoINASolver(
            instance=instance,
            network=network,
            K=K,
            jobs_num=jobs_num,
            Cs=Cs,
            Ps=Ps,
            topo_name='FatTree',
            solver_name='gurobi',
            time_limit=300,
            mip_gap=0.01,
            verbose=True  # 改为True以显示求解过程
        )
        
        milp_solver.build_model()
        solution = milp_solver.solve()
        m_runtime = time.time() - m_start

        print("-"*80)

        if solution is not None:
            # solve() returns the extracted solution dict
            m_makespan = solution['makespan']
            
            print(f"  ✓ Makespan: {m_makespan:.4f}")
            print(f"  ✓ Runtime: {m_runtime:.4f}s")
            
            milp_results.append({
                'makespan': m_makespan,
                'runtime': m_runtime,
                'success': True,
                'placement': solution['ina_placement_switches']
            })
            
            # Calculate comparison metrics
            gap = ((h_makespan - m_makespan) / m_makespan) * 100
            speedup = h_runtime / m_runtime if m_runtime > 0 else float('inf')
            
            print(f"\n[Comparison]")
            print(f"  • Optimality gap: {gap:.2f}%")
            print(f"  • Heuristic speedup: {speedup:.2f}x")
            print(f"  • Makespan ratio (H/M): {h_makespan/m_makespan:.4f}")
        else:
            print("-"*80)
            print(f"  ✗ Failed to find solution")
            print(f"  ✓ Runtime: {m_runtime:.4f}s")
            milp_results.append({
                'makespan': float('inf'),
                'runtime': m_runtime,
                'success': False
            })
    except Exception as e:
        m_runtime = time.time() - m_start
        print("-"*80)
        print(f"  ✗ Error: {e}")
        print(f"  ✓ Runtime: {m_runtime:.4f}s")
        milp_results.append({
            'makespan': float('inf'),
            'runtime': m_runtime,
            'success': False
        })

# ========== Summary Statistics ==========
print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")

h_makespans = [r['makespan'] for r in heuristic_results if r['makespan'] < float('inf')]
h_runtimes = [r['runtime'] for r in heuristic_results]

m_makespans = [r['makespan'] for r in milp_results if r['success'] and r['makespan'] < float('inf')]
m_runtimes = [r['runtime'] for r in milp_results if r['success']]

if h_makespans:
    print(f"\n🎯 Heuristic Algorithm:")
    print(f"  • Avg Makespan: {np.mean(h_makespans):.4f} ± {np.std(h_makespans):.4f}")
    print(f"  • Min/Max: {np.min(h_makespans):.4f} / {np.max(h_makespans):.4f}")
    print(f"  • Avg Runtime: {np.mean(h_runtimes):.4f}s")

if m_makespans:
    print(f"\n🔧 MILP Solver:")
    print(f"  • Avg Makespan: {np.mean(m_makespans):.4f} ± {np.std(m_makespans):.4f}")
    print(f"  • Min/Max: {np.min(m_makespans):.4f} / {np.max(m_makespans):.4f}")
    print(f"  • Avg Runtime: {np.mean(m_runtimes):.4f}s")
    print(f"  • Success Rate: {len(m_makespans)}/{instances_num} ({100*len(m_makespans)/instances_num:.1f}%)")

if h_makespans and m_makespans:
    gaps = [((h_makespans[i] - m_makespans[i]) / m_makespans[i]) * 100 
            for i in range(min(len(h_makespans), len(m_makespans)))]
    speedups = [h_runtimes[i] / m_runtimes[i] 
                for i in range(min(len(h_runtimes), len(m_runtimes)))]
    
    print(f"\n📈 Comparison:")
    print(f"  • Avg Optimality Gap: {np.mean(gaps):.2f}% ± {np.std(gaps):.2f}%")
    print(f"  • Min/Max Gap: {np.min(gaps):.2f}% / {np.max(gaps):.2f}%")
    print(f"  • Avg Speedup: {np.mean(speedups):.2f}x")
    
    avg_gap = np.mean(gaps)
    if avg_gap < 5:
        print(f"  ✓ Heuristic is near-optimal (< 5% gap)")
    elif avg_gap < 10:
        print(f"  ~ Heuristic is good (< 10% gap)")
    else:
        print(f"  ⚠ Heuristic has significant gap (> 10%)")

print(f"\n{'='*80}")
print("✓ Comparison completed!")
print(f"{'='*80}")























