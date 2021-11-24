from multiprocessing import Pool
from pathlib import Path

import numpy as np
import toml
from loguru import logger

from Config import Model
from aedtproject import AEDTProject
from aedtproject.OptConfig import OptConfig
from aedtproject.algorithms import EM_NSGAII
from aedtproject.optproblem import OptProblem
from aedtproject.run_ansys_exe import run_ansysedt_script
from core.MagneticFieldData import MagneticFieldData

config_file = r'../configs/dataset_gen.toml'
config_dict = toml.load(config_file)
config = OptConfig(**config_dict)

export_config_file = r'../configs/config_unet_MAE.toml'
export_conf = Model(**toml.load(export_config_file))
project = AEDTProject(config.aedtProject,
                      config.design_name)


def run_export(output_dir, timestamp, export_config: Model):
    logger.info(timestamp)
    field_dir = output_dir / str(timestamp)
    field_dir.mkdir(exist_ok=True)
    bxy_filename = field_dir / 'bxy.fld'
    CD_filename = field_dir / 'current_density.fld'
    material_filename = field_dir / 'material.fld'
    paras = {
        'project_path': (
                output_dir / f'{output_dir.name}.aedt').absolute().as_posix(),
        'project_name': output_dir.name,
        'design_name': export_config.project.active_design_name,
        'bxy_filename': bxy_filename.absolute().as_posix(),
        'current_density_filename': CD_filename.absolute().as_posix(),
        'material_filename': material_filename.absolute().as_posix(),
        'time': timestamp,
        'time_unit': export_config.export.time_uint,
        'xmin': export_config.export.xmin,
        'xmax': export_config.export.xmax,
        'xstep': export_config.export.xstep,
        'ymin': export_config.export.ymin,
        'ymax': export_config.export.ymax,
        'ystep': export_config.export.ystep,
        'unit': export_config.export.unit
    }
    export_script_str = export_config.export.script_start
    export_script_str += export_config.export.scripts.format(**paras)
    script_obj = field_dir / 'export.py'
    script_obj.write_text(export_script_str, encoding='utf-8')
    if not material_filename.exists():
        run_ansysedt_script(
            ansysedt_exe_path=Path(config.Simulation.ansysedt_path),
            script=script_obj,
            timeout_in_sec=config.Simulation.Timeout_in_minutes * 60)


total_time_steps = 234
force = []


def export_single_file(output_dir):
    for i in range(total_time_steps):
        # output_dir.mkdir(exist_ok=True)
        run_export(output_dir, i, export_conf)
        # output_dir = Path(config.project.dataset_dir) / str(i)

        data = MagneticFieldData(output_dir / f'{i}', export_conf)
        # data.plot_all('all.png', outpath=output_dir, suffix='.png')
        force.append(data.get_force())

    np.save(output_dir / 'force.npy', force)


def generate_models(totalsize=20):
    problem = OptProblem(config, project)
    algorithm = EM_NSGAII(problem=problem, population_size=totalsize)
    algorithm.initialize()


if __name__ == '__main__':
    # step 1:
    # generate_models(20)
    # step 2:
    parent_path = Path(
        r'C:\Users\TF\Documents\models\dataset')
    locks = parent_path.glob('**/*.aedt.lock')
    [lock.unlink() for lock in locks]
    dirs = [parent_path / f'{i}' for i in range(1, 21)]
    with Pool(processes=8) as pool:
        pool.map(export_single_file, dirs)
