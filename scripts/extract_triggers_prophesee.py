import argparse
from enum import Enum
from pathlib import Path
import numpy as np
from metavision_core.event_io import RawReader, EventsIterator

class TriggerSource(Enum):
    FB_CAMERA = 1
    EXTERNAL = 2

def get_ext_trigger_timestamps(rawfile: Path):
    assert rawfile.exists()
    assert rawfile.suffix == ".raw"

    mv_iterator = EventsIterator(str(rawfile), delta_t=10000, max_duration=None)
    all_triggers = []
    all_polarities = []
    for evs in mv_iterator:
        if evs.size != 0:
            triggers = mv_iterator.reader.get_ext_trigger_events()
            if len(triggers) > 0:
                for trigger in triggers:
                    all_triggers.append(trigger[1])
                    all_polarities.append(trigger[0])
                    # all_triggers.append({'time': trigger[1], 'polarity': trigger[0]})
                mv_iterator.reader.clear_ext_trigger_events()

    # rawreader = RawReader(str(rawfile))
    # print(rawreader)
    # print(str(rawfile))

    # # while not rawreader.is_done():
    # #     print(rawreader.is_done())
    # #     arr = rawreader.load_delta_t(10**5)
    # #     print(arr)
    # ext_trigger_list = rawreader.get_ext_trigger_events()

    time = np.array(all_triggers)
    pol = np.array(all_polarities)

    return time, pol

def get_reconstruction_timestamps(time: np.ndarray, pol: np.ndarray, trigger_source: TriggerSource, time_offset_us: int=0):
    assert 0 <= pol.max() <= 1
    assert np.all(np.abs(np.diff(pol)) == 1), 'polarity must alternate from trigger to trigger'

    timestamps = None
    if trigger_source == TriggerSource.FB_CAMERA:
        assert pol[0] == 1, 'first ext trigger polarity must be positive'
        assert pol[-1] == 0, 'last ext trigger polarity must be negative'
        # rising_ts = time[pol==1]
        falling_ts = time[pol==0]
        timestamps = falling_ts.astype('int64')
    else:
        timestamps = time[pol==1]
        timestamps = timestamps + time_offset_us

    return timestamps

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Read trigger data from prophesee raw file')
    parser.add_argument('rawfile')
    parser.add_argument('output_file', help='Path to output text file with timestamps for reconstruction.')
    parser.add_argument('--fb-source', '-fbs', action='store_true',
            help='The frame-based camera is the trigger source.')
    parser.add_argument('--external-source', '-es', action='store_true',
            help='An external trigger source is used.')
    parser.add_argument('--time_offset_us', '-offset', type=int, default=0,
            help='Add a constant time offset to the timestamp for reconstruction (Only for external source).'
            )

    args = parser.parse_args()

    rawfile = Path(args.rawfile)
    outfile = Path(args.output_file)
    assert not outfile.exists()
    assert outfile.suffix == '.txt'

    trigger_source = TriggerSource.FB_CAMERA
    if args.external_source:
        trigger_source = TriggerSource.EXTERNAL

    times, polarities = get_ext_trigger_timestamps(rawfile)
    reconstruction_ts = get_reconstruction_timestamps(times, polarities, trigger_source, args.time_offset_us)
    np.savetxt(str(outfile), reconstruction_ts, '%i')
