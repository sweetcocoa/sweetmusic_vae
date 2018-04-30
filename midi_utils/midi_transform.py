import pretty_midi
from sweetmusic_vae.midi_utils.reverse_pianoroll import *
import random
import numpy as np

LEN_MIDI_ROLL = 1500


def get_piano_roll(pm):
    return pm.get_piano_roll()


def random_crop_midi(pm_roll):
    """
    piano roll 인스턴스를 받아서 일정한 시간축 길이(LEN_MIDI_ROLL)로 자르고 그 잘린 객체를 반환합니다. 자르는 구간은 랜덤으로 정해집니다.
    만일 일정한 시간축 길이보다 더 짧은 roll이 들어오면, repeat으로 시간을 늘린 다음 객체를 자릅니다.
    :param pm_roll:
    :return:
    """
    length = len(pm_roll[0])
    if length < LEN_MIDI_ROLL:
        print("Warning :: Too short midi to crop")
        pm_new = np.concatenate((pm_roll, pm_roll), axis=1)
        return random_crop_midi(pm_new)
    else:
        From = random.randint(0, length - LEN_MIDI_ROLL)
        To = From + LEN_MIDI_ROLL
        pm_new = pm_roll[:, From:To]
        return pm_new
    pass

if __name__ == "__main__":
    # print(random.randint(0, 1))
    a = np.zeros((10,99))
    print(random_crop_midi(a).shape)
