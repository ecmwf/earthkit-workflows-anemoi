from abc import abstractmethod, ABCMeta
import logging
from functools import cached_property

import earthkit.data as ekd
from earthkit.data import settings

LOG = logging.getLogger(__name__)

class RequestBasedInput(metaclass=ABCMeta):
    def __init__(self, checkpoint, dates):
        self.checkpoint = checkpoint
        self.dates = dates

    WHERE: str # Location of the data source

    @abstractmethod
    def pl_load_source(self, **kwargs): ...
    @abstractmethod
    def sfc_load_source(self, **kwargs): ...
    @abstractmethod
    def ml_load_source(self, **kwargs): ...

    @cached_property
    def fields_sfc(self):
        param = self.checkpoint.param_sfc
        if not param:
            return ekd.from_source("empty")

        LOG.info(f"Loading surface fields from {self.WHERE}")
        return ekd.from_source(
            "multi",
            [
                self.sfc_load_source(
                    date=date,
                    time=time,
                    param=param,
                    grid=self.checkpoint.grid,
                    area=self.checkpoint.area,
                )
                for date, time in self.dates
            ],
        )

    @cached_property
    def fields_pl(self):
        param, level = self.checkpoint.param_level_pl
        if not (param and level):
            return ekd.from_source("empty")

        LOG.info(f"Loading pressure fields from {self.WHERE}")
        return ekd.from_source(
            "multi",
            [
                self.pl_load_source(
                    date=date,
                    time=time,
                    param=param,
                    level=level,
                    grid=self.checkpoint.grid,
                    area=self.checkpoint.area,
                )
                for date, time in self.dates
            ],
        )

    @cached_property
    def fields_ml(self):
        param, level = self.checkpoint.param_level_ml
        if not (param and level):
            return ekd.from_source("empty")

        LOG.info(f"Loading model fields from {self.WHERE}")
        return ekd.from_source(
            "multi",
            [
                self.ml_load_source(
                    date=date,
                    time=time,
                    param=param,
                    level=level,
                    grid=self.checkpoint.grid,
                    area=self.checkpoint.area,
                )
                for date, time in self.dates
            ],
        )

    @cached_property
    def all_fields(self):
        return self.fields_sfc + self.fields_pl + self.fields_ml


class MarsInput(RequestBasedInput):
    WHERE = "MARS"

    def pl_load_source(self, **kwargs):
        kwargs["levtype"] = "pl"
        logging.debug("load source mars %s", kwargs)
        return ekd.from_source("mars", kwargs, log = None)

    def sfc_load_source(self, **kwargs):
        kwargs["levtype"] = "sfc"
        logging.debug("load source mars %s", kwargs)
        return ekd.from_source("mars", kwargs, log = None)

    def ml_load_source(self, **kwargs):
        kwargs["levtype"] = "ml"
        logging.debug("load source mars %s", kwargs)
        return ekd.from_source("mars", kwargs, log = None)

class FileInput(RequestBasedInput):
    WHERE = "file"

    def pl_load_source(self, **kwargs):
        kwargs["levtype"] = "pl"
        logging.debug("load source file %s", kwargs)
        return ekd.from_source("file", kwargs)

    def sfc_load_source(self, **kwargs):
        kwargs["levtype"] = "sfc"
        logging.debug("load source file %s", kwargs)
        return ekd.from_source("file", kwargs)

    def ml_load_source(self, **kwargs):
        kwargs["levtype"] = "ml"
        logging.debug("load source file %s", kwargs)
        return ekd.from_source("file", kwargs)
