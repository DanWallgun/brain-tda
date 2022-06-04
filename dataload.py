import glob
import os
import pickle
import re
import typing
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class Record:
    subject_id: str
    session_id: str
    run_id: str
    atlas_name: str
    patient: bool
    data: typing.Any 
    laplacian_eigvals: np.array = None

    @property
    def full_id(self) -> typing.Tuple[str, str, str]:
        return (self.subject_id, self.session_id, self.run_id)


@dataclass
class AtlasToData:
    aal: pd.DataFrame = None
    basc: pd.DataFrame = None
    msdl: pd.DataFrame = None


@dataclass
class MultiRecord:
    subject_id: str
    session_id: str
    run_id: str
    patient: bool
    atlas_to_data: AtlasToData = AtlasToData()

    @property
    def data(self) -> pd.DataFrame:
        return self.atlas_to_data.basc


file_min_size = 15 << 10


def load_records(pkl_path: str, ts_extracted_path: str = None) -> typing.List[Record]:
    if ts_extracted_path is not None:
        path_regex = re.compile(f'{ts_extracted_path}/(?P<group_type>[^/]+)s/[^/]+/sub\-(?P<subject_id>.+)_ses\-d(?P<session_id>.+)_task\-rest_run\-(?P<run_id>.+)_atlas\-(?P<atlas_name>.+)\.csv')
        def parse_record(path: str) -> Record:
            info = path_regex.match(path)
            assert info
            info = info.groupdict()
            assert info['group_type'] in ['patient', 'control']
            patient = info.pop('group_type') == 'patient'
            return Record(
                patient=patient,
                data=pd.read_csv(path, index_col=0),
                **info
            )

        records = []
        for path in tqdm(glob.glob(f'{ts_extracted_path}/*/*/*.csv')):
            if os.path.getsize(path) < file_min_size:
                continue
            records.append(parse_record(path))

        pickle.dump(records, open(pkl_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        records: typing.List[Record] = pickle.load(open(pkl_path, 'rb'))
    return records


def load_multirecords(pkl_path: str, ts_extracted_path: str = None) -> typing.List[MultiRecord]:
    records = load_records(pkl_path, ts_extracted_path)
    multirecords: typing.Dict[typing.Any, MultiRecord] = dict()
    for record in records:
        multirecord = multirecords.get(record.full_id)
        if multirecord is not None:
            multirecord.atlas_to_data[record.atlas_name] = record.data
        else:
            multirecords[record.full_id] = MultiRecord(
                record.subject_id,
                record.session_id,
                record.run_id,
                record.patient, 
                AtlasToData(**{record.atlas_name: record.data})
            )
