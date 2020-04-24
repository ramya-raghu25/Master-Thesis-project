"""
DESCRIPTION:     Python file for compressing images
COPYRIGHT:       © 2020 Robert Bosch GmbH

The reproduction, distribution and utilization of this file as
well as the communication of its contents to others without express
authorization is prohibited. Offenders will be held liable for the
payment of damages. All rights reserved in the event of the grant
of a patent, utility model or design.

A major portion of the code was taken from the following link:
https://stackoverflow.com/questions/45699932/asyncio-stdout-failing
Date: 20.10.2019

"""

from pathlib import Path
import asyncio, sys , click
from tqdm import tqdm
import logging

if sys.platform == 'win32':
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)

log = logging.getLogger('Compressing images')
log.setLevel(logging.DEBUG)


def async_map(func, tasks, num_concurrent=4):
    num_tasks = tasks.__len__()

    queue = asyncio.Queue()
    for idx, task in enumerate(tasks): queue.put_nowait((idx, task))

    res = [None] * num_tasks

    pbar = tqdm(total=num_tasks)

    async def async_worker():
        while not queue.empty():
            idx, task = queue.get_nowait()

            result = await func(task)
            res[idx] = result
            pbar.update(1)

            queue.task_done()

    joint_future = asyncio.gather(
        *(async_worker() for i in range(num_concurrent))
    )

    asyncio.get_event_loop().run_until_complete(joint_future)
    pbar.close()
    return res


async def run_external_program(cmd):
    try:
        proc = await asyncio.create_subprocess_exec(
            *map(str, cmd),  # str to convert Path objects
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
    except Exception as e:
        cmd_as_str = ' '.join(map(str, cmd))
        log.exception(f'run_externam_program({cmd_as_str})')
        raise e

    if proc.returncode != 0:
        cmd_as_str = ' '.join(map(str, cmd))
        log.error(f"""Command {cmd_as_str} error, retcode = {proc.returncode}
--Stderr--
{stderr}
--Stdout--
{stdout}""")

@click.command()
@click.argument('source', type=click.Path(exists=True, dir_okay=True))
@click.argument('destination', type=click.Path())
@click.argument('cmd', type=str)
@click.option('--ext', default='.webp')
@click.option('--concurrent', default=20)
def main(source, destination, cmd, ext, concurrent):
    print(source, destination, cmd, ext, concurrent)
    source = Path(source)
    destination = Path(destination)
    print(source, destination)
    log.info('Loading images')
    src_files = list(source.glob('**/*.png'))
    dest_files = [destination / p.relative_to(source) for p in src_files]

    dirs_to_create = set(p.parent for p in dest_files)
    log.info(f'Creating new {dirs_to_create.__len__()} dirs')
    for par in dirs_to_create:
        par.mkdir(parents=True, exist_ok=True)

    cmds = [
        cmd.format(src=src_file, dest=dest_file.with_suffix(ext)).split()
        for (src_file, dest_file) in zip(src_files, dest_files)
    ]
    async_map(run_external_program, cmds, num_concurrent=concurrent)


if __name__ == '__main__':
    main()