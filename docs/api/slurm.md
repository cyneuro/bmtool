# SLURM API Reference

This page provides API reference documentation for the SLURM module, which contains functions and classes for managing batch simulations on SLURM-based HPC clusters.

## Utility Functions

::: bmtool.SLURM.check_job_status
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.SLURM.submit_job
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.SLURM.send_teams_message
    options:
      show_root_heading: true
      heading_level: 3

## Parameter Sweep Classes

::: bmtool.SLURM.seedSweep
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - edit_json
        - change_json_file_path

::: bmtool.SLURM.multiSeedSweep
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - edit_all_jsons

## Simulation Block Management

::: bmtool.SLURM.SimulationBlock
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - create_batch_script
        - submit_block
        - check_block_status
        - check_block_completed
        - check_block_running
        - check_block_submited

## File Transfer

::: bmtool.SLURM.get_relative_path
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.SLURM.globus_transfer
    options:
      show_root_heading: true
      heading_level: 3

## BlockRunner

::: bmtool.SLURM.BlockRunner
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - submit_blocks_sequentially
        - submit_blocks_parallel
