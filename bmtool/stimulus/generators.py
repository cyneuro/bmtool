import numpy as np


def get_stim_cycle(on_time, off_time, t_start=0.0, t_stop=10.0, verbose=False):
    """Get burst input stimulus parameters, (duration, number) of cycles.
    Poisson input is first on for on_time, starting at t_start, then off for
    off_time. This repeats until the last on_time can complete before t_stop.
    """
    t_cycle = on_time + off_time
    n_cycle = int(np.floor((t_stop + off_time - t_start) / t_cycle))
    return t_cycle, n_cycle


def get_fr_short(n_assemblies, firing_rate=(0., 0., 0.),
                 on_time=1.0, off_time=0.5,
                 t_start=0.0, t_stop=10.0, n_cycles=None, n_rounds=1, verbose=False,
                 assembly_index=None):
    """Short burst is delivered to each assembly sequentially within each cycle.
    
    Args:
        n_assemblies: Total number of assemblies.
        firing_rate: 3-tuple of firing rates (off_rate, burst_fr, silent_rate)
            firing_rate[0] = off_rate (background rate during non-burst on-time)
            firing_rate[1] = burst_fr (burst firing rate)
            firing_rate[2] = silent_rate (rate during off-time and before t_start)
        on_time: Duration of on period (s)
        off_time: Duration of off period (s)
        t_start: Start time of the stimulus cycles (s)
        t_stop: Stop time of the stimulus cycles (s)
        n_rounds: Number of short bursts each assembly receives per cycle.
            Can be fractional; some assemblies will receive one more burst per cycle.
        verbose: If True, print detailed information
        assembly_index: List of selected assembly indices. If provided, generates traces for all
                       n_assemblies. Non-selected assemblies fire at off_rate during on_time.
        
    Returns:
        list: Firing rate traces for each assembly
    """
    # Set assembly_index to all if not provided
    if assembly_index is None:
        assembly_index = list(range(n_assemblies))
    
    # In generalized version, n_assemblies passed IS the total assemblies.
    total_assemblies = n_assemblies
    
    if verbose:
        print("\nStarting get_fr_short...")
        print(f"Selected assemblies: {assembly_index} out of {total_assemblies}")
        print(f"Firing rates (off, burst, silent): {firing_rate}")
        print(f"Time parameters - start: {t_start}, stop: {t_stop}, on_time: {on_time}, off_time: {off_time}")
        print(f"Rounds per cycle: {n_rounds}")

    # Ensure firing_rate is properly formatted as 3-tuple
    firing_rate = np.asarray(firing_rate).ravel()[:3]
    if len(firing_rate) < 3:
        # Pad with zeros if insufficient values provided
        firing_rate = np.concatenate((firing_rate, np.zeros(3 - firing_rate.size)))
    
    off_rate, burst_fr, silent_rate = firing_rate
    assembly_index = list(assembly_index)  # Ensure it's a list

    t_cycle = on_time + off_time
    if n_cycles is not None:
        n_cycle = n_cycles
    else:
        n_cycle = int((t_stop - t_start) // t_cycle)
    
    # Calculate burst timing within each cycle (based on selected assemblies only)
    n_selected = len(assembly_index)
    if n_selected == 0:
        n_bursts_per_cycle = 0
    else:
        n_bursts_per_cycle = int(np.ceil(n_rounds * n_selected))
    n_rounds_int = int(np.ceil(n_rounds))
    
    if verbose:
        print(f"\nCycle information:")
        print(f"Time per cycle: {t_cycle}")
        print(f"Number of cycles: {n_cycle}")
        print(f"Bursts per cycle: {n_bursts_per_cycle}")
        print(f"Rounds (integer): {n_rounds_int}")

    # Calculate time slots for bursts within on_time
    if n_bursts_per_cycle > 0:
        burst_duration = on_time / n_bursts_per_cycle
        burst_times = np.linspace(0, on_time - burst_duration, n_bursts_per_cycle)
    else:
        burst_times = []

    params = []
    
    for i in range(total_assemblies):
        if verbose:
            print(f"\nProcessing assembly {i}...")
            
        # Initialize with silent rate from 0 to t_start
        times = [0.0, t_start]
        rates = [silent_rate, silent_rate]
        
        is_selected = i in assembly_index
        
        for cycle in range(n_cycle):
            cycle_start = t_start + cycle * t_cycle
            on_period_start = cycle_start
            on_period_end = cycle_start + on_time
            cycle_end = cycle_start + t_cycle
            
            if verbose and cycle == 0:
                print(f"  Cycle {cycle}: start={cycle_start}, on_end={on_period_end}, cycle_end={cycle_end}")
            
            if is_selected:
                # Determine which bursts this assembly gets in this cycle
                # Map assembly i to its position in assembly_index
                try:
                    selected_position = assembly_index.index(i)
                except ValueError:
                    selected_position = -1 # Should not happen if is_selected checked
                
                assembly_bursts = []
                for round_num in range(n_rounds_int):
                    burst_index = round_num * n_selected + selected_position
                    if burst_index < n_bursts_per_cycle:
                        burst_start_time = on_period_start + burst_times[burst_index]
                        burst_end_time = burst_start_time + burst_duration
                        # Make sure burst doesn't exceed on_period
                        burst_end_time = min(burst_end_time, on_period_end)
                        assembly_bursts.append((burst_start_time, burst_end_time))
                
                if verbose and cycle == 0:
                    print(f"  Assembly {i} gets {len(assembly_bursts)} bursts in cycle {cycle}")
                
                # Add timepoints for this cycle
                current_time = on_period_start
                
                # Handle the on period with bursts
                for burst_start, burst_end in assembly_bursts:
                    # Before burst (background rate)
                    if current_time < burst_start:
                        times.extend([current_time, current_time, burst_start, burst_start])
                        rates.extend([silent_rate, off_rate, off_rate, silent_rate])
                    
                    # During burst (burst rate)
                    times.extend([burst_start, burst_start, burst_end, burst_end])
                    rates.extend([silent_rate, burst_fr, burst_fr, silent_rate])
                    
                    current_time = burst_end
                
                # After all bursts until end of on period (background rate)
                if current_time < on_period_end:
                    times.extend([current_time, current_time, on_period_end, on_period_end])
                    rates.extend([silent_rate, off_rate, off_rate, silent_rate])
            else:
                # Non-selected assembly: fires at off_rate during on_time
                times.extend([on_period_start, on_period_start, on_period_end, on_period_end])
                rates.extend([silent_rate, off_rate, off_rate, silent_rate])
            
            # Off period (silent rate)
            times.extend([on_period_end, on_period_end, cycle_end, cycle_end])
            rates.extend([silent_rate, silent_rate, silent_rate, silent_rate])
        
        # Add final timepoint
        times.append(t_stop)
        rates.append(silent_rate)
        
        params.append({
            'firing_rate': rates,
            'times': times
        })
            
    return params


def get_fr_long(n_assemblies, firing_rate=(0., 0., 0.),
                on_time=1.0, off_time=0.5,
                t_start=0.0, t_stop=10.0, n_cycles=None, verbose=False,
                assembly_index=None):
    """Long burst where one assembly is active per cycle.
    
    Args:
        n_assemblies (int): Total number of assemblies.
        firing_rate (tuple): 3-tuple (off_rate, burst_rate, silent_rate).
        on_time (float): Duration of active cycle (s).
        off_time (float): Duration of silent period (s).
        t_start (float): Start time (s).
        t_stop (float): Stop time (s).
        n_cycles (int, optional): Number of cycles. Defaults to floor of duration.
        verbose (bool): Whether to print debug info.
        assembly_index (list, optional): List of assemblies to generate traces for.
        
    Returns:
        list: Firing rate parameters for each assembly.
    """
    if assembly_index is None:
        assembly_index = list(range(n_assemblies))
    
    total_assemblies = n_assemblies
    
    if verbose:
        print("\nStarting get_fr_long...")
        print(f"Selected assemblies: {assembly_index} out of {total_assemblies}")
        print(f"Firing rates (off, burst, silent): {firing_rate}")
        print(f"Time parameters - start: {t_start}, stop: {t_stop}, on_time: {on_time}, off_time: {off_time}")

    # Ensure firing_rate is properly formatted as 3-tuple
    firing_rate = np.asarray(firing_rate).ravel()[:3]
    if len(firing_rate) < 3:
        # Pad with zeros if insufficient values provided
        firing_rate = np.concatenate((firing_rate, np.zeros(3 - firing_rate.size)))
    
    off_rate, burst_fr, silent_rate = firing_rate
    assembly_index = list(assembly_index)  # Ensure it's a list
    
    t_cycle = on_time + off_time
    if n_cycles is not None:
        n_cycle = n_cycles
    else:
        n_cycle = int((t_stop - t_start) // t_cycle)
    n_selected = len(assembly_index)

    if verbose:
        print(f"\nCycle information:")
        print(f"Time per cycle: {t_cycle}")
        print(f"Number of cycles: {n_cycle}")

    params = []
    
    for i in range(total_assemblies):
        # Initialize with silent rate from 0 to t_start
        times = [0.0, t_start]
        rates = [silent_rate, silent_rate]
        
        is_selected = i in assembly_index
        
        for cycle in range(n_cycle):
            cycle_start = t_start + cycle * t_cycle
            burst_start = cycle_start
            burst_end = cycle_start + on_time
            cycle_end = cycle_start + t_cycle
            
            # Map to position in selected assemblies only
            if n_selected > 0:
                active_position = cycle % n_selected
                active_assembly = assembly_index[active_position]
            else:
                active_assembly = -1
            
            if i == active_assembly:
                # This assembly is active - burst_fr during on_time, silent during off_time
                times.extend([
                    burst_start,
                    burst_start,
                    burst_end,
                    burst_end,
                    cycle_end,
                    cycle_end
                ])
                rates.extend([
                    silent_rate,  # At start of burst
                    burst_fr,  # During burst
                    burst_fr,  # During burst
                    silent_rate,  # End of burst
                    silent_rate,  # During off time
                    silent_rate   # Until next cycle
                ])
            elif is_selected:
                # This assembly is selected but inactive - off_rate during on_time
                times.extend([
                    burst_start,
                    burst_start,
                    burst_end,
                    burst_end,
                    cycle_end,
                    cycle_end
                ])
                rates.extend([
                    silent_rate,  # At start of burst
                    off_rate,  # During active assembly's burst
                    off_rate,  # During active assembly's burst
                    silent_rate,  # End of burst
                    silent_rate,  # During off time
                    silent_rate   # Until next cycle
                ])
            else:
                # Non-selected assembly: always silent or background (use off_rate during on_time)
                times.extend([
                    burst_start,
                    burst_start,
                    burst_end,
                    burst_end,
                    cycle_end,
                    cycle_end
                ])
                rates.extend([
                    silent_rate,  # At start of burst
                    off_rate,  # Background during on_time
                    off_rate,  # Background during on_time
                    silent_rate,  # End of burst
                    silent_rate,  # During off time
                    silent_rate   # Until next cycle
                ])
        
        # Add final timepoint
        times.append(t_stop)
        rates.append(silent_rate)
        
        params.append({
            'firing_rate': rates,
            'times': times
        })
            
    return params


def get_fr_ramp(n_assemblies, firing_rate=(0., 0., 0., 0.),
                on_time=1.0, off_time=0.5,
                ramp_on_time=None, ramp_off_time=None,
                t_start=0.0, t_stop=10.0, n_cycles=None, verbose=False,
                assembly_index=None):
    """Ramping input where one assembly is active per cycle, with linear rate changes.
    
    Args:
        n_assemblies (int): Total number of assemblies.
        firing_rate (tuple): 4-tuple (off_rate, ramp_start_fr, ramp_end_fr, silent_rate).
        on_time (float): Duration of active cycle (s).
        off_time (float): Duration of silent period (s).
        ramp_on_time (float, optional): Offset within on_time to start ramp.
        ramp_off_time (float, optional): Offset within on_time to end ramp.
        t_start (float): Start time (s).
        t_stop (float): Stop time (s).
        n_cycles (int, optional): Number of cycles.
        verbose (bool): Whether to print debug info.
        assembly_index (list, optional): List of assemblies to generate traces for.
        
    Returns:
        list: Firing rate parameters for each assembly.
    """
    if assembly_index is None:
        assembly_index = list(range(n_assemblies))
    
    total_assemblies = n_assemblies
    
    if verbose:
        print("\nStarting get_fr_ramp...")
        print(f"Selected assemblies: {assembly_index} out of {total_assemblies}")
        print(f"Firing rates (off, ramp_start, ramp_end, silent): {firing_rate}")
        print(f"Time parameters - start: {t_start}, stop: {t_stop}, on_time: {on_time}, off_time: {off_time}")

    # Ensure firing_rate is properly formatted
    firing_rate = np.asarray(firing_rate).ravel()[:4]
    firing_rate = np.concatenate((np.zeros(4 - firing_rate.size), firing_rate))

    off_rate = firing_rate[0]
    ramp_start_fr = firing_rate[1]
    ramp_end_fr = firing_rate[2]
    silent_rate = firing_rate[3]
    assembly_index = list(assembly_index)  # Ensure it's a list

    # Set ramp timing within on_time
    ramp_off_time = on_time if ramp_off_time is None else min(ramp_off_time, on_time)
    ramp_on_time = 0. if ramp_on_time is None else min(ramp_on_time, ramp_off_time)

    t_cycle = on_time + off_time
    if n_cycles is not None:
        n_cycle = n_cycles
    else:
        n_cycle = int((t_stop - t_start) // t_cycle)
    n_selected = len(assembly_index)

    if verbose:
        print(f"\nCycle information:")
        print(f"Time per cycle: {t_cycle}")
        print(f"Number of cycles: {n_cycle}")
        print(f"Ramp timing: {ramp_on_time} to {ramp_off_time} within on_time of {on_time}")

    params = []
    
    for i in range(total_assemblies):
        # Initialize with zero rate from 0 to t_start
        times = [0.0, t_start]
        rates = [silent_rate, silent_rate]
        
        is_selected = i in assembly_index
        
        for cycle in range(n_cycle):
            cycle_start = t_start + cycle * t_cycle
            on_period_start = cycle_start
            on_period_end = cycle_start + on_time
            cycle_end = cycle_start + t_cycle
            
            # Map to position in selected assemblies only
            if n_selected > 0:
                active_position = cycle % n_selected
                active_assembly = assembly_index[active_position]
            else:
                active_assembly = -1
            
            if i == active_assembly:
                # This assembly is active - ramping pattern during on_time, silent during off_time
                
                # Before ramp starts (constant at ramp_start_fr)
                if ramp_on_time > 0:
                    times.extend([on_period_start, on_period_start, 
                                on_period_start + ramp_on_time, on_period_start + ramp_on_time])
                    rates.extend([silent_rate, ramp_start_fr, ramp_start_fr, silent_rate])
                
                # During ramp (linear increase from ramp_start_fr to ramp_end_fr)
                times.extend([on_period_start + ramp_on_time, on_period_start + ramp_on_time,
                            on_period_start + ramp_off_time, on_period_start + ramp_off_time])
                rates.extend([silent_rate, ramp_start_fr, ramp_end_fr, silent_rate])
                
                # After ramp ends (constant at ramp_end_fr)
                if ramp_off_time < on_time:
                    times.extend([on_period_start + ramp_off_time, on_period_start + ramp_off_time,
                                on_period_end, on_period_end])
                    rates.extend([silent_rate, ramp_end_fr, ramp_end_fr, silent_rate])
                
                # Off period (silent rate)
                times.extend([on_period_end, on_period_end, cycle_end, cycle_end])
                rates.extend([silent_rate, silent_rate, silent_rate, silent_rate])
                
            elif is_selected:
                # This assembly is selected but inactive - off_rate during on_time
                times.extend([
                    on_period_start,
                    on_period_start,
                    on_period_end,
                    on_period_end,
                    cycle_end,
                    cycle_end
                ])
                rates.extend([
                    silent_rate,  # At start of active assembly's burst
                    off_rate,  # During active assembly's on_time (background rate)
                    off_rate,  # During active assembly's on_time (background rate)
                    silent_rate,  # End of on_time
                    silent_rate,  # During off time
                    silent_rate   # Until next cycle
                ])
            else:
                # Non-selected assembly: fires at off_rate during on_time
                times.extend([
                    on_period_start,
                    on_period_start,
                    on_period_end,
                    on_period_end,
                    cycle_end,
                    cycle_end
                ])
                rates.extend([
                    silent_rate,  # At start
                    off_rate,  # Background during on_time
                    off_rate,  # Background during on_time
                    silent_rate,  # End of on_time
                    silent_rate,  # During off time
                    silent_rate   # Until next cycle
                ])
        
        # Add final timepoint
        times.append(t_stop)
        rates.append(silent_rate)
        
        params.append({
            'firing_rate': rates,
            'times': times
        })
            
    return params


def get_fr_join(n_assemblies, firing_rate=(0., 0., 0.),
                on_time=1.0, off_time=0.5,
                quit=False, ramp_on_time=None, ramp_off_time=None,
                t_start=0.0, t_stop=10.0, n_cycles=None, n_steps=20, verbose=False,
                assembly_index=None):
    """Input is delivered to an increasing portion of one assembly in each cycle.
    
    This function generates multiple parameter sets per assembly (controlled by n_steps),
    simulating a gradual recruitment ('join') or withdrawal ('quit') of neurons.
    
    Args:
        n_assemblies (int): Total number of assemblies.
        firing_rate (tuple): 3-tuple (off_rate, on_rate, silent_rate).
        on_time (float): Duration of active cycle (s).
        off_time (float): Duration of silent period (s).
        quit (bool): If True, neurons start on and quit. If False, join one by one.
        ramp_on_time (float, optional): Offset within on_time to start recruitment.
        ramp_off_time (float, optional): Offset within on_time to end recruitment.
        t_start (float): Start time (s).
        t_stop (float): Stop time (s).
        n_steps (int): Number of steps (neuron subgroups) within each assembly.
        assembly_index (list, optional): List of selected assembly indices.
        
    Returns:
        list of dict: Firing rate parameters, including 'assembly' and 'step' metadata.
    """
    if assembly_index is None:
        assembly_index = list(range(n_assemblies))
    
    total_assemblies = n_assemblies
    
    if verbose:
        print("\nStarting get_fr_join...")
        print(f"Selected assemblies: {assembly_index} out of {total_assemblies}")
        print(f"Firing rates (off, on, silent): {firing_rate}")
        print(f"n_steps: {n_steps}, quit mode: {quit}")

    # Ensure firing_rate is properly formatted
    firing_rate = np.asarray(firing_rate).ravel()[:3]
    firing_rate = np.concatenate((np.zeros(3 - firing_rate.size), firing_rate))

    off_rate = firing_rate[0]
    on_rate = firing_rate[1]
    silent_rate = firing_rate[2]
    assembly_index = list(assembly_index)  # Ensure it's a list

    # Set recruitment timing within on_time
    ramp_off_time = on_time if ramp_off_time is None else min(ramp_off_time, on_time)
    ramp_on_time = 0. if ramp_on_time is None else min(ramp_on_time, ramp_off_time)

    t_cycle = on_time + off_time
    if n_cycles is not None:
        n_cycle = n_cycles
    else:
        n_cycle = int((t_stop - t_start) // t_cycle)
    n_selected = len(assembly_index)

    # Calculate step timing offsets (when each step gets recruited)
    t_offset = np.linspace(ramp_on_time, ramp_off_time, n_steps, endpoint=False)
    if quit:
        t_offset = t_offset[::-1]

    if verbose:
        print(f"Cycle information:")
        print(f"Time per cycle: {t_cycle}")
        print(f"Number of cycles: {n_cycle}")
        print(f"Recruitment timing: {ramp_on_time} to {ramp_off_time}")
        print(f"Step times: {t_offset}")

    # Generate one parameter set per assembly
    # Each selected assembly will have n_steps sub-groups that use this same pattern
    # but applied to different neuron groups (handled externally)
    all_params = []
    
    for assy_idx in range(total_assemblies):
        is_selected = assy_idx in assembly_index
        
        if is_selected:
            # Selected assembly: generate n_steps parameter sets (one for each neuron sub-group)
            for step_idx, step_time in enumerate(t_offset):
                # Initialize with silent rate from 0 to t_start
                times = [0.0, t_start]
                rates = [silent_rate, silent_rate]
                
                for cycle in range(n_cycle):
                    cycle_start = t_start + cycle * t_cycle
                    on_period_start = cycle_start
                    on_period_end = cycle_start + on_time
                    cycle_end = cycle_start + t_cycle
                    
                    # Map to position in selected assemblies only
                    if n_selected > 0:
                        active_position = cycle % n_selected
                        active_assembly = assembly_index[active_position]
                    else:
                        active_assembly = -1
                    
                    if assy_idx == active_assembly:
                        # This assembly is active in this cycle
                        # This specific step gets recruited at step_time
                        
                        if quit:
                            # Quit mode: start with on_rate, switch to silent at step_time
                            recruit_time = on_period_start + step_time
                            
                            # Before quit time (on_rate)
                            if step_time > 0:
                                times.extend([on_period_start, on_period_start,
                                            recruit_time, recruit_time])
                                rates.extend([silent_rate, on_rate, on_rate, silent_rate])
                            
                            # After quit time (silent)
                            times.extend([recruit_time, recruit_time,
                                        on_period_end, on_period_end])
                            rates.extend([silent_rate, silent_rate, silent_rate, silent_rate])
                        else:
                            # Join mode: start silent, switch to on_rate at step_time
                            recruit_time = on_period_start + step_time
                            
                            # Before recruit time (silent)
                            if step_time > 0:
                                times.extend([on_period_start, on_period_start,
                                            recruit_time, recruit_time])
                                rates.extend([silent_rate, silent_rate, silent_rate, silent_rate])
                            
                            # After recruit time (on_rate)
                            times.extend([recruit_time, recruit_time,
                                        on_period_end, on_period_end])
                            rates.extend([silent_rate, on_rate, on_rate, silent_rate])
                        
                        # Off period (silent)
                        times.extend([on_period_end, on_period_end, cycle_end, cycle_end])
                        rates.extend([silent_rate, silent_rate, silent_rate, silent_rate])
                        
                    else:
                        # This selected assembly is inactive - fire at off_rate during on_time
                        times.extend([
                            on_period_start,
                            on_period_start,
                            on_period_end,
                            on_period_end,
                            cycle_end,
                            cycle_end
                        ])
                        rates.extend([
                            silent_rate,  # At start of on_time
                            off_rate,  # During active assembly's on_time (background rate)
                            off_rate,  # During active assembly's on_time (background rate)
                            silent_rate,  # End of on_time
                            silent_rate,  # During off time
                            silent_rate   # Until next cycle
                        ])
                
                # Add final timepoint
                times.append(t_stop)
                rates.append(silent_rate)
                
                all_params.append({
                    'firing_rate': rates,
                    'times': times,
                    'assembly': assy_idx,
                    'step': step_idx
                })
        else:
            # Non-selected assembly: single parameter set (fires at off_rate during all on_times)
            times = [0.0, t_start]
            rates = [silent_rate, silent_rate]
            
            for cycle in range(n_cycle):
                cycle_start = t_start + cycle * t_cycle
                on_period_start = cycle_start
                on_period_end = cycle_start + on_time
                cycle_end = cycle_start + t_cycle
                
                # Always fire at off_rate during on_time, silent during off_time
                times.extend([
                    on_period_start,
                    on_period_start,
                    on_period_end,
                    on_period_end,
                    cycle_end,
                    cycle_end
                ])
                rates.extend([
                    silent_rate,  # At start of on_time
                    off_rate,  # Background during on_time
                    off_rate,  # Background during on_time
                    silent_rate,  # End of on_time
                    silent_rate,  # During off time
                    silent_rate   # Until next cycle
                ])
            
            # Add final timepoint
            times.append(t_stop)
            rates.append(silent_rate)
            
            all_params.append({
                'firing_rate': rates,
                'times': times,
                'assembly': assy_idx,
                'step': None
            })
    
    return all_params


def get_fr_fade(n_assemblies, off_rate=10., firing_rate=(0., 0., 0., 0.),
                on_time=1.0, off_time=0.5,
                ramp_on_time=None, ramp_off_time=None,
                t_start=0.0, t_stop=10.0, n_cycles=None, verbose=False,
                assembly_index=None):
    """Input fades in and out between a pair of assemblies in each cycle.
    
    Args:
        n_assemblies: Total number of assemblies
        off_rate: firing rate of assemblies not involved in current fade cycle
        firing_rate: 4-tuple of firing rates (fade_out_start, fade_out_end, fade_in_start, fade_in_end)
        on_time, off_time: on / off time durations
        ramp_on_time, ramp_off_time: start and end time of ramp in on time duration
        t_start, t_stop: start and stop time of the stimulus cycles
        verbose: if True, print detailed information
        assembly_index: List of selected assembly indices.
    
    Return: list of firing rate parameter dictionaries
    """
    if assembly_index is None:
        assembly_index = list(range(n_assemblies))
    
    total_assemblies = n_assemblies
    
    if verbose:
        print("\nStarting get_fr_fade...")
        print(f"Selected assemblies: {assembly_index} out of {total_assemblies}")
        print(f"Firing rates (fade_out_start, fade_out_end, fade_in_start, fade_in_end): {firing_rate}")
        print(f"Off rate for non-active assemblies: {off_rate}")
        print(f"Time parameters - start: {t_start}, stop: {t_stop}, on_time: {on_time}, off_time: {off_time}")
    
    # Ensure firing_rate is properly formatted
    firing_rate = np.asarray(firing_rate).ravel()[:4]
    if firing_rate.size < 4:
        firing_rate = np.concatenate((np.zeros(4 - firing_rate.size), firing_rate))
    fade_out_start, fade_out_end, fade_in_start, fade_in_end = firing_rate
    assembly_index = list(assembly_index)  # Ensure it's a list
    
    # Set ramp timing within on_time
    ramp_off_time = on_time if ramp_off_time is None else min(ramp_off_time, on_time)
    ramp_on_time = 0. if ramp_on_time is None else min(ramp_on_time, ramp_off_time)
    
    # Calculate cycle parameters
    t_cycle = on_time + off_time
    if n_cycles is not None:
        n_cycle = n_cycles
    else:
        n_cycle = int((t_stop - t_start) // t_cycle)
    silent_rate = 0.0
    n_selected = len(assembly_index)
    
    if verbose:
        print(f"\nCycle information:")
        print(f"Time per cycle: {t_cycle}")
        print(f"Number of cycles: {n_cycle}")
        print(f"Ramp timing: {ramp_on_time} to {ramp_off_time} within on_time of {on_time}")
    
    params = []
    
    for i in range(total_assemblies):
        # Initialize with zero rate from 0 to t_start
        times = [0.0, t_start]
        rates = [silent_rate, silent_rate]
        
        is_selected = i in assembly_index
        
        for cycle in range(n_cycle):
            cycle_start = t_start + cycle * t_cycle
            on_period_start = cycle_start
            on_period_end = cycle_start + on_time
            cycle_end = cycle_start + t_cycle
            
            # Determine which pair is active
            n_pairs = n_selected // 2 if n_selected >= 2 else 0
            
            if n_pairs > 0:
                pair_idx = cycle % n_pairs
                if pair_idx * 2 + 1 < n_selected:
                    fade_out_assembly = assembly_index[pair_idx * 2]
                    fade_in_assembly = assembly_index[pair_idx * 2 + 1]
                else:
                    fade_out_assembly = -1
                    fade_in_assembly = -1
            else:
                fade_out_assembly = -1
                fade_in_assembly = -1
            
            if i == fade_out_assembly:
                # Fading out
                if ramp_on_time > 0:
                    times.extend([on_period_start, on_period_start,
                                on_period_start + ramp_on_time, on_period_start + ramp_on_time])
                    rates.extend([silent_rate, fade_out_start, fade_out_start, silent_rate])
                
                times.extend([on_period_start + ramp_on_time, on_period_start + ramp_on_time,
                            on_period_start + ramp_off_time, on_period_start + ramp_off_time])
                rates.extend([silent_rate, fade_out_start, fade_out_end, silent_rate])
                
                if ramp_off_time < on_time:
                    times.extend([on_period_start + ramp_off_time, on_period_start + ramp_off_time,
                                on_period_end, on_period_end])
                    rates.extend([silent_rate, fade_out_end, fade_out_end, silent_rate])
                
                times.extend([on_period_end, cycle_end, cycle_end])
                rates.extend([silent_rate, silent_rate, silent_rate])
                
            elif i == fade_in_assembly:
                # Fading in
                if ramp_on_time > 0:
                    times.extend([on_period_start, on_period_start,
                                on_period_start + ramp_on_time, on_period_start + ramp_on_time])
                    rates.extend([silent_rate, fade_in_start, fade_in_start, silent_rate])
                
                times.extend([on_period_start + ramp_on_time, on_period_start + ramp_on_time,
                            on_period_start + ramp_off_time, on_period_start + ramp_off_time])
                rates.extend([silent_rate, fade_in_start, fade_in_end, silent_rate])
                
                if ramp_off_time < on_time:
                    times.extend([on_period_start + ramp_off_time, on_period_start + ramp_off_time,
                                on_period_end, on_period_end])
                    rates.extend([silent_rate, fade_in_end, fade_in_end, silent_rate])
                
                times.extend([on_period_end, cycle_end, cycle_end])
                rates.extend([silent_rate, silent_rate, silent_rate])
                
            else:
                # Inactive assembly: fires at off_rate during active pair's on_time
                times.extend([
                    on_period_start,
                    on_period_start,
                    on_period_end,
                    on_period_end,
                    cycle_end,
                    cycle_end
                ])
                rates.extend([
                    silent_rate,
                    off_rate,
                    off_rate,
                    silent_rate,
                    silent_rate,
                    silent_rate
                ])
        
        # Add final timepoint
        times.append(t_stop)
        rates.append(silent_rate)
        
        params.append({
            'firing_rate': rates,
            'times': times
        })
    
    return params


def get_fr_loop(n_assemblies, firing_rate=(0., 0., 0.),
                on_times=(1.0, ), off_time=0.5,
                t_start=0.0, t_stop=10.0, verbose=False):
    """Poisson input is first on for on_time starting at t_start, then off for
    off_time.
    """
    firing_rate = np.asarray(firing_rate).ravel()
    on_times = np.fmax(np.sort(np.asarray(on_times).ravel()), 0)
    if on_times[0]:
        on_times = np.insert(on_times, 0, 0.)
    if firing_rate.size - on_times.size != 1:
        raise ValueError("Length of `firing_rate` should be len(on_times) + 1.")
    t_cycle, n_cycle = get_stim_cycle(on_times[-1], off_time, t_start, t_stop)

    times = [[0] for _ in range(n_assemblies)]
    for j in range(n_cycle):
        ts = t_start + t_cycle * j + on_times
        times[j % n_assemblies].extend(np.insert(ts, [0, -1], ts[[0, -1]]))

    params = []
    fr = []
    fr0 = firing_rate[0]
    for ts in times:
        ts.append(t_stop)
        n = (len(ts) - 2) // (on_times.size + 2)
        if len(fr) != len(ts):
            fr = np.append(np.tile(np.insert(firing_rate, 0, fr0), n), [fr0, fr0])
        params.append(dict(firing_rate=fr, times=ts))
    return params
