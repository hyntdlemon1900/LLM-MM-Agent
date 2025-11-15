import random
import random
from typing import List, Dict


def generate_dataset_job_random_fattree(jobs_num: int, all_workers_id: List[int], num_instances: int, num_edge_switches: int, sequential_allocation: bool = False) -> Dict:
    """
    Generate a dataset with multiple instances for a fat-tree topology, each containing worker and PS allocations for jobs.
    Workers are contiguously assigned to edge switches (workers_per_switch workers per switch), and all workers are 
    allocated to jobs either sequentially (contiguous ranges) or randomly, with one worker randomly selected as the PS 
    per job. Worker IDs for each job are sorted before being added. Includes the number of workers per edge switch for 
    each job (excluding PS).

    Args:
        jobs_num (int): Number of jobs per instance.
        all_workers_id (List[int]): List of worker IDs (e.g., [0, 1, ..., n-1]).
        num_instances (int): Number of instances to generate.
        num_edge_switches (int): Number of edge switches in the topology.
        sequential_allocation (bool): If True, allocate workers sequentially (contiguous ranges); if False, allocate randomly.

    Returns:
        Dict: Dataset with instances, each containing 'workers_id', 'ps_id', 'workers_num', 'jobs_size',
              and 'tor_switch_workers' mapping edge switches to worker counts per job (excluding PS).
    """
    random.seed(1)  # Ensure reproducibility
    dataset = {}
    total_servers = len(all_workers_id)

    # Validate input: workers_per_switch must be evenly distributed across edge switches
    if total_servers % num_edge_switches != 0 or jobs_num <= 0 or jobs_num > total_servers:
        return {}
    workers_per_switch = total_servers // num_edge_switches
    if workers_per_switch <= 0:
        return {}

    # Minimum job size (including PS) and maximum size for variance
    min_size = 10
    avg_size = total_servers // jobs_num
    max_size = min(total_servers - min_size * (jobs_num - 1), avg_size * 2)

    for name in range(num_instances):
        instance = {
            'workers_id': [],
            'ps_id': [],
            'workers_num': [],
            'jobs_size': [],
            'tor_switch_workers': []
        }

        # Assign workers contiguously to edge switches (0-19 to switch 0, 20-39 to switch 1, etc.)
        edge_switch_assignments = [all_workers_id[i * workers_per_switch:(i + 1) * workers_per_switch] 
                                  for i in range(num_edge_switches)]

        # Map worker IDs to their assigned edge switch for quick lookup
        worker_to_switch = {}
        for switch_idx, switch_workers in enumerate(edge_switch_assignments):
            for worker in switch_workers:
                worker_to_switch[worker] = switch_idx

        # Generate job sizes to use all 160 workers
        job_sizes = []
        remaining_workers = total_servers
        for i in range(jobs_num - 1):
            size = random.randint(min_size, min(max_size, remaining_workers - min_size * (jobs_num - i - 1)))
            job_sizes.append(size)
            remaining_workers -= size
        job_sizes.append(max(min_size, remaining_workers))
        random.shuffle(job_sizes)

        # Allocate workers to jobs
        used_servers = set()
        remaining_workers = all_workers_id.copy()  # Copy for random selection
        current_worker_idx = 0  # For sequential allocation
        for job_idx, size in enumerate(job_sizes):
            # Select workers for the job
            if len(remaining_workers) < size:
                instance['workers_id'].append([])
                instance['ps_id'].append(-1)
                instance['workers_num'].append(0)
                instance['jobs_size'].append(0)
                instance['tor_switch_workers'].append({})
                continue

            if sequential_allocation:
                # Sequential allocation: take contiguous range of workers
                job_workers = all_workers_id[current_worker_idx:current_worker_idx + size]
                current_worker_idx += size
            else:
                # Random allocation: select random workers
                job_workers = random.sample(remaining_workers, size)

            if any(w in used_servers for w in job_workers):
                instance['workers_id'].append([])
                instance['ps_id'].append(-1)
                instance['workers_num'].append(0)
                instance['jobs_size'].append(0)
                instance['tor_switch_workers'].append({})
                continue

            # Randomly select one worker as the PS
            ps = random.choice(job_workers)
            final_job_workers = [w for w in job_workers if w != ps]

            # Sort worker IDs directly (as integers)
            final_job_workers = sorted(final_job_workers)

            # Assign random computational job size
            computational_job_size = random.randint(10, 1000)

            # Count workers (excluding PS) per edge switch for this job
            switch_worker_counts = {i: 0 for i in range(num_edge_switches)}
            for worker in final_job_workers:  # Only count workers, exclude PS
                switch_idx = worker_to_switch[worker]
                switch_worker_counts[switch_idx] += 1

            # Update instance data
            instance['workers_id'].append(final_job_workers)
            instance['ps_id'].append(ps)
            instance['workers_num'].append(len(final_job_workers))
            instance['jobs_size'].append(computational_job_size)
            instance['tor_switch_workers'].append(switch_worker_counts)

            # Mark servers as used and remove from remaining workers
            used_servers.update(job_workers)
            remaining_workers = [w for w in remaining_workers if w not in job_workers]

        dataset[f'instance_{name}'] = instance

    return dataset

def generate_dataset_job_random_spineleaf(jobs_num: int, all_workers_id: List[int], num_instances: int, num_edge_switches: int, sequential_allocation: bool = False) -> Dict:
    """
    Generate a dataset with multiple instances for a spine-leaf topology, each containing worker and PS allocations for jobs.
    Workers are contiguously assigned to edge switches (workers_per_switch workers per switch), and all workers are 
    allocated to jobs either sequentially (contiguous ranges) or randomly, with one worker randomly selected as the PS 
    per job. Worker IDs for each job are sorted before being added. Includes the number of workers per edge switch for 
    each job (excluding PS).

    Args:
        jobs_num (int): Number of jobs per instance.
        all_workers_id (List[int]): List of worker IDs (e.g., [0, 1, ..., n-1]).
        num_instances (int): Number of instances to generate.
        num_edge_switches (int): Number of edge switches in the topology.
        sequential_allocation (bool): If True, allocate workers sequentially (contiguous ranges); if False, allocate randomly.

    Returns:
        Dict: Dataset with instances, each containing 'workers_id', 'ps_id', 'workers_num', 'jobs_size',
              and 'tor_switch_workers' mapping edge switches to worker counts per job (excluding PS).
    """
    random.seed(1)  # Ensure reproducibility
    dataset = {}
    total_servers = len(all_workers_id)

    # Validate input: workers_per_switch must be evenly distributed across edge switches
    if total_servers % num_edge_switches != 0 or jobs_num <= 0 or jobs_num > total_servers:
        return {}
    workers_per_switch = total_servers // num_edge_switches
    if workers_per_switch <= 0:
        return {}

    # Minimum job size (including PS) and maximum size for variance
    min_size = 10
    avg_size = total_servers // jobs_num
    max_size = min(total_servers - min_size * (jobs_num - 1), avg_size * 2)

    for name in range(num_instances):
        instance = {
            'workers_id': [],
            'ps_id': [],
            'workers_num': [],
            'jobs_size': [],
            'tor_switch_workers': []
        }

        # Assign workers contiguously to edge switches (0-19 to switch 0, 20-39 to switch 1, etc.)
        edge_switch_assignments = [all_workers_id[i * workers_per_switch:(i + 1) * workers_per_switch] 
                                  for i in range(num_edge_switches)]

        # Map worker IDs to their assigned edge switch for quick lookup
        worker_to_switch = {}
        for switch_idx, switch_workers in enumerate(edge_switch_assignments):
            for worker in switch_workers:
                worker_to_switch[worker] = switch_idx

        # Generate job sizes to use all 160 workers
        job_sizes = []
        remaining_workers = total_servers
        for i in range(jobs_num - 1):
            size = random.randint(min_size, min(max_size, remaining_workers - min_size * (jobs_num - i - 1)))
            job_sizes.append(size)
            remaining_workers -= size
        job_sizes.append(max(min_size, remaining_workers))
        random.shuffle(job_sizes)

        # Allocate workers to jobs
        used_servers = set()
        remaining_workers = all_workers_id.copy()  # Copy for random selection
        current_worker_idx = 0  # For sequential allocation
        for job_idx, size in enumerate(job_sizes):
            # Select workers for the job
            if len(remaining_workers) < size:
                instance['workers_id'].append([])
                instance['ps_id'].append(-1)
                instance['workers_num'].append(0)
                instance['jobs_size'].append(0)
                instance['tor_switch_workers'].append({})
                continue

            if sequential_allocation:
                # Sequential allocation: take contiguous range of workers
                job_workers = all_workers_id[current_worker_idx:current_worker_idx + size]
                current_worker_idx += size
            else:
                # Random allocation: select random workers
                job_workers = random.sample(remaining_workers, size)

            if any(w in used_servers for w in job_workers):
                instance['workers_id'].append([])
                instance['ps_id'].append(-1)
                instance['workers_num'].append(0)
                instance['jobs_size'].append(0)
                instance['tor_switch_workers'].append({})
                continue

            # Randomly select one worker as the PS
            ps = random.choice(job_workers)
            final_job_workers = [w for w in job_workers if w != ps]

            # Sort worker IDs directly (as integers)
            final_job_workers = sorted(final_job_workers)

            # Assign random computational job size
            computational_job_size = random.randint(10, 1000)

            # Count workers (excluding PS) per edge switch for this job
            switch_worker_counts = {i: 0 for i in range(num_edge_switches)}
            for worker in final_job_workers:  # Only count workers, exclude PS
                switch_idx = worker_to_switch[worker]
                switch_worker_counts[switch_idx] += 1

            # Update instance data
            instance['workers_id'].append(final_job_workers)
            instance['ps_id'].append(ps)
            instance['workers_num'].append(len(final_job_workers))
            instance['jobs_size'].append(computational_job_size)
            instance['tor_switch_workers'].append(switch_worker_counts)

            # Mark servers as used and remove from remaining workers
            used_servers.update(job_workers)
            remaining_workers = [w for w in remaining_workers if w not in job_workers]

        dataset[f'instance_{name}'] = instance

    return dataset

def generate_dataset_job_ft(jobs_num: int, all_workers_id: List[int], num_instances: int, num_edge_switches: int, sequential_allocation: bool = False) -> Dict:
    random.seed(1)
    dataset_new = {}
    
    if jobs_num > 6:
        dataset = generate_dataset_job_random_fattree(jobs_num, all_workers_id, num_instances, num_edge_switches, sequential_allocation)
        dataset_new = dataset
    else:
        dataset = generate_dataset_job_random_fattree(6, all_workers_id, num_instances, num_edge_switches, sequential_allocation)
        for name in dataset:
            instance = dataset[name]
            lst = [i for i in range(6)]
            lst = random.sample(lst, jobs_num)

            instance_new = {
                'workers_id': [],
                'ps_id': [],
                'workers_num': [],
                'jobs_size':[],
                'tor_switch_workers':[]
            }

            instance_new['workers_id'] = [instance['workers_id'][i] for i in lst]
            instance_new['ps_id'] = [instance['ps_id'][i] for i in lst]
            instance_new['workers_num'] = [instance['workers_num'][i] for i in lst]
            instance_new['jobs_size'] = [instance['jobs_size'][i] for i in lst]
            instance_new['tor_switch_workers'] = [instance['tor_switch_workers'][i] for i in lst]
            dataset_new[f'instance_{name}'] = instance_new
    return dataset_new

def generate_dataset_job_sl(jobs_num: int, all_workers_id: List[int], num_instances: int, num_edge_switches: int, sequential_allocation: bool = False) -> Dict:
    random.seed(1)
    dataset_new = {}
    
    if jobs_num > 6:
        dataset = generate_dataset_job_random_spineleaf(jobs_num, all_workers_id, num_instances, num_edge_switches, sequential_allocation)
        dataset_new = dataset
    else:
        dataset = generate_dataset_job_random_spineleaf(6, all_workers_id, num_instances, num_edge_switches, sequential_allocation)
        for name in dataset:
            instance = dataset[name]
            lst = [i for i in range(6)]
            lst = random.sample(lst, jobs_num)

            instance_new = {
                'workers_id': [],
                'ps_id': [],
                'workers_num': [],
                'jobs_size':[],
                'tor_switch_workers':[]
            }

            instance_new['workers_id'] = [instance['workers_id'][i] for i in lst]
            instance_new['ps_id'] = [instance['ps_id'][i] for i in lst]
            instance_new['workers_num'] = [instance['workers_num'][i] for i in lst]
            instance_new['jobs_size'] = [instance['jobs_size'][i] for i in lst]
            instance_new['tor_switch_workers'] = [instance['tor_switch_workers'][i] for i in lst]
            dataset_new[f'instance_{name}'] = instance_new
    return dataset_new