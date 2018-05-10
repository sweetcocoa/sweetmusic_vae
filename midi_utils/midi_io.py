import pretty_midi
import librosa
import IPython.display
from reverse_pianoroll import piano_roll_to_pretty_midi


def show_midi(pm, roll=False):
    """
    Piano roll 이나 PrettyMidi 인스턴스를 jupyter notebook상의 셀에서 재생 가능한 형태로 보여줍니다.
    (Show Piano Roll or PrettyMidi instances that can be played in a cell in the jupyter notebook.)

    :param pm:
    :param roll:
        parameter인 pm이 Piano roll인지를 나타냅니다. 꼭 필요한 것은 아닙니다.
        (Indicates whether the parameter pm is a Piano roll. It is not necessary.)
        Default : False
    :return:
        None.
    """
    if not roll and isinstance(pm, pretty_midi.PrettyMIDI):
        IPython.display.display(IPython.display.Audio(pm.synthesize(fs=16000), rate=16000))
    elif roll:
        new_pm = piano_roll_to_pretty_midi(pm)
        IPython.display.display(IPython.display.Audio(new_pm.synthesize(fs=16000), rate=16000))
    else:
        print("show_midi Warning : unexpected arguments")
        new_pm = piano_roll_to_pretty_midi(pm)
        IPython.display.display(IPython.display.Audio(new_pm.synthesize(fs=16000), rate=16000))


def save_pm(pm, path):
    pm.write(path)


def is_single_track(pm):
    if len(pm.instruments) != 1:
        return False
    else:
        return True


def get_sliding_windows(pm_, WINDOW_LENGTH=16, WINDOW_STRIDE=16):
    """
    piano roll이나 PrettyMidi 인스턴스를 정해진 윈도우 길이로 슬라이딩하며 분할합니다.
    Magenta VAE 논문에서는 window length : 2 bar, srtide 1 bar 로 sequence를 분할했습니다.

    :return:
        list[PrettyMidi, PrettyMidi ... PrettyMidi]
    """
    pm = None

    if isinstance(pm_ , pretty_midi.PrettyMIDI):
        pm = pm_.get_piano_roll()
    else:
        pm = pm_

    windows = []
    len_pm = len(pm[0])
    for i in range(0, len_pm, WINDOW_STRIDE):
        if len_pm - i < WINDOW_LENGTH:
            break
        pm_piece = piano_roll_to_pretty_midi(pm[:, i:i+WINDOW_LENGTH])
        windows.append(pm_piece)
    
    return windows


def check_tempo(pm, numerator=4, denominator=4):
    """
        4/4 박자 midi인지 확인합니다.
        
    """
    if len(pm.time_signature_changes) != 1:
        return False
    nume = pm.time_signature_changes[0].numerator
    denom = pm.time_signature_changes[0].denominator
    if nume == numerator and denom == denominator:
        return True
    else:
        return False


def is_piano_track(pm):
    '''
    PrettyMidi 클래스의 인스턴스가 Piano Track인지를 확인합니다.
    (Check that the Pretty Midi instance is a Piano Track.)
    Parameters
    ----------
    pm : PrettyMidi

    Returns
    -------
    ret : Bool
        pm이 피아노 트랙이라면 True, 아니라면 False를 반환합니다.
        (return True if pm is a piano track, False otherwise.)
    '''

    if is_single_track(pm):
        return not pm.instruments[0].is_drum
    else:
        for inst in pm.instruments:
            if inst.is_drum:
                return False
        return True


def is_available_track(pm):
    ret = is_piano_track(pm) and check_tempo(pm) and is_single_track(pm)
    return ret
