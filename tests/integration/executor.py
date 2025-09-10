# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import random
import time
from multiprocessing import Process
from typing import Any

import cascade.gateway.api as api
import cascade.gateway.client as client
from cascade.gateway.server import serve
from cascade.low import views as cascade_views
from cascade.low.core import JobInstance
from earthkit.workflows.fluent import Action

from earthkit.workflows import Cascade

logger = logging.getLogger(__name__)


def spawn_gateway(max_jobs: int | None = None) -> tuple[str, Process]:
    url = f"tcp://localhost:{random.randint(12000, 32000)}"
    p = Process(target=serve, args=(url,), kwargs={"max_jobs": max_jobs})
    p.start()
    return url, p


def convert_to_job(action: Action) -> JobInstance:
    from cascade.low.into import graph2job

    graph = Cascade(action.graph())
    job = graph2job(graph._graph)
    sinks = cascade_views.sinks(job)
    job.ext_outputs = list(sink for sink in sinks)
    return job


def convert_to_jobspec(job: JobInstance, *, workers_per_host: int = 1) -> api.JobSpec:
    return api.JobSpec(
        envvars={},
        workers_per_host=workers_per_host,
        hosts=1,
        benchmark_name=None,
        use_slurm=False,
        job_instance=job,
    )


def run_job(action: Action, *, url: str, tries: int = 16) -> Any:
    """Run a job and return the result."""
    job = convert_to_job(action)

    for task in job.tasks:  # DISABLE GPU REQUIREMENT
        if task.startswith("run_as_earthkit"):
            job.tasks[task].definition.needs_gpu = False

    job_spec = convert_to_jobspec(job)

    submit_job_req = api.SubmitJobRequest(job=job_spec)
    submit_job_res: api.SubmitJobResponse = client.request_response(submit_job_req, url=url)  # type: ignore
    job_id = submit_job_res.job_id

    assert submit_job_res.error is None
    assert job_id is not None

    result = get_result(job, job_id=job_id, url=url, tries=tries)

    return result


def get_result(job: JobInstance, job_id, url: str, *, tries=16) -> Any:
    """Get result of a job from the gateway"""

    job_progress_req = api.JobProgressRequest(job_ids=[job_id])

    while tries > 0:
        job_progress_res: api.JobProgressResponse = client.request_response(job_progress_req, url)
        assert job_progress_res.error is None
        is_computed = job_progress_res.progresses[job_id].pct == "100.00"
        is_datasets = job.ext_outputs[0] in job_progress_res.datasets[job_id]
        if is_computed and is_datasets:
            break
        else:
            if tries % 5:
                logger.info(f"{tries}: Current progress is {job_progress_res}")
            if job_progress_res.progresses[job_id].started:
                tries -= 1
            time.sleep(1)
    assert tries > 0, "Exhausted all tries waiting for job to complete"

    result_retrieval_req = api.ResultRetrievalRequest(job_id=job_id, dataset_id=job.ext_outputs[0])

    result_retrieval_res = client.request_response(result_retrieval_req, url)
    assert result_retrieval_res.error is None
    assert result_retrieval_res.result is not None

    deser = api.decoded_result(result_retrieval_res, job)
    return deser
