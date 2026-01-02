import time
import minorminer
import dwave.inspector
from dwave.samplers import SimulatedAnnealingSampler
from hybrid.reference.kerberos import KerberosSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite, LeapHybridSampler
from dwave.embedding.chain_strength import uniform_torque_compensation


def perform_optical_flow(bqm, mode, rows, cols):
    if mode == 'c':
        print("Performing Dense Optical Flow...")
        # Simulated Annealing Process
        start_time = time.time()
        sampleset = SimulatedAnnealingSampler().sample(bqm, label="Dense Optical Flow",
                                                       beta_range=[.1, 25.0], beta_schedule_type='linear', num_reads=1500, num_sweeps=750)
        # sampleset = SimulatedAnnealingSampler().sample(bqm)
        # sampleset = SimulatedAnnealingSampler().sample(
        #     bqm, label="Dense Optical Flow", num_reads=50)
        # decoded_samples = model.decode_sampleset(sampleset)
        best_sample = sampleset.first
        print("--- %s seconds ---" % (time.time() - start_time))
    elif mode == 'hq':
        # Hybrid Quantum Approach
        print("Performing Hybrid Quantum Optical Flow...")
        start_time = time.time()
        # https://docs.ocean.dwavesys.com/projects/system/en/stable/reference/generated/dwave.system.samplers.LeapHybridSampler.sample.html
        # https://docs.ocean.dwavesys.com/projects/system/en/stable/reference/samplers.html#leaphybridsampler
        # https://docs.dwavesys.com/docs/latest/doc_leap_hybrid.html
        sampleset = LeapHybridSampler().sample(bqm, label='Hybrid Quantum Optical Flow')
        print(sampleset.info)

        # https://docs.ocean.dwavesys.com/en/stable/docs_hybrid/reference/reference.html

        # sampleset = KerberosSampler().sample(bqm, max_iter=10, convergence=5, qpu_reads=200,
        #                                      qpu_params={'label': 'Hybrid Quantum Optical Flow'})
        best_sample = sampleset.first
        print("--- %s seconds ---" % (time.time() - start_time))
    elif mode == 'q':  # Use with SMALL PROBLEMS!
        chain_str_offset = 0.5
        chain_strength = -1
        num_runs = 1500
        print("Embedding QUBO in QPU and performing Quantum Optical Flow...")
        # Quantum Annealing Process
        sampler = DWaveSampler(
            region="eu-central-1", solver={"name": 'Advantage_system5.4', "topology__type": 'pegasus'})
        # sampler = DWaveSampler(region="na-west-1", solver={"name": 'Advantage_system4.1', "topology__type": 'pegasus'})
        # sampler = DWaveSampler(region="na-west-1", solver={"name": 'Advantage_system6.4', "topology__type": 'pegasus'})
        # sampler = DWaveSampler(region="na-west-1", solver={"name": 'Advantage2_prototype2.3', "topology__type": 'zephyr'})

        # sampler = DWaveSampler()
        # https://docs.dwavesys.com/docs/latest/c_qpu_annealing.html
        print("Maximum anneal schedule points: {}".format(
            sampler.properties["max_anneal_schedule_points"]))
        annealing_range = sampler.properties["annealing_time_range"]
        max_slope = 1.0/annealing_range[0]
        print("Annealing time range: {}".format(
            sampler.properties["annealing_time_range"]))

        print("Maximum slope:", max_slope)

        schedule = [[0.0, 0.0], [10.0, 0.5], [11, 1.0]]
        print("Schedule: %s" % schedule)
        src_graph = list(bqm.quadratic.keys())

        target_graph = sampler.edgelist
        start_time = time.time()
        embedding, found = minorminer.find_embedding(
            src_graph, target_graph, return_overlap=True)
        print(found)
        # https://github.com/FarinaMatteo/qmmf/blob/master/problems/disjoint_set_cover.py
        # choose the chain strength according to the maximum chain length criterion
        if chain_strength == -1:
            chain_strength = max(
                list(len(chain) for chain in embedding.values())) + chain_str_offset
        elif chain_strength is None:
            chain_strength = uniform_torque_compensation(bqm, embedding)
        print("Running with chain strength: ({}x{})".format(
            rows, cols), chain_strength)

        # instantiate the sampler and run it to find low energy states of the possible instantiations
        sampler = FixedEmbeddingComposite(sampler, embedding=embedding)

        sampleset = sampler.sample(bqm, num_reads=num_runs, chain_strength=chain_strength,
                                   return_embedding=True, anneal_schedule=schedule, label="Quantum Optical Flow")

        # solution = sampler.sample(bqm, chain_strength=chain_strength, num_reads=num_runs, label="Quantum Optical Flow")
        best_sample = sampleset.first
        print("--- %s seconds ---" % (time.time() - start_time))
        # Show inspector if needed
        # dwave.inspector.show(sampleset)

    return best_sample
