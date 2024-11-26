from __future__ import annotations

from datetime import datetime
from typing import Any, Generator, TYPE_CHECKING

from earthkit.data import FieldList, SimpleFieldList, ArrayField

from anemoi.utils.dates import frequency_to_seconds

if TYPE_CHECKING:
    from anemoi.inference.runner import Runner

def run(input_state: dict, runner: Runner, lead_time: int) -> Generator[Any, None, None]:
    """
    Run the model

    Parameters
    ----------
    runner : Runner
        Runner object
    input_state : dict
        Initial conditions for the model
    lead_time : int
        Lead time for the model

    Yields
    ------
    Generator[SimpleFieldList, None, None]
        State of the model at each time step
    """
    yield from runner.run(input_state=input_state, lead_time=lead_time)

def run_as_earthkit(input_state: dict, runner: Runner, lead_time: int) -> Generator[SimpleFieldList, None, None]:
    """
    Run the model and yield the results as earthkit FieldList

    Parameters
    ----------
    runner : Runner
        Runner Object
    input_state : dict
        Initial Conditions for the model
    lead_time : int
        Lead time for the model

    Yields
    ------
    Generator[SimpleFieldList, None, None]
        State of the model at each time step
    """    
    initial_date: datetime = input_state['date']
    for state in run(input_state, runner, lead_time):
        fields = []
        step = frequency_to_seconds(state['date'] - initial_date) // 3600
        
        for field in state['fields']:
            fields.append(ArrayField(state['fields'][field], {'param': field, 'step': step, 'base_datetime': initial_date}))

        yield FieldList.from_fields(fields)

def collect_as_earthkit(input_state: dict, runner: Runner, lead_time: int) -> SimpleFieldList:
    """
    Collect the results of the model run as earthkit FieldList

    Parameters
    ----------
    runner : Runner
        Runner object
    input_state : dict
        Initial conditions for the model
    lead_time : int
        Lead time for the model

    Returns
    -------
    SimpleFieldList
        Combined FieldList of the model run
    """
    fields = []
    for state in run_as_earthkit(input_state, runner, lead_time):
        fields.extend(state.fields)
    return SimpleFieldList(fields)