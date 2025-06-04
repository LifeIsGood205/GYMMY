import os
import re

import Settings as s






if __name__ == "__main__":
    s.audio_path = "C:/Users/Administrator/Gymmy/audio files/Hebrew/Female/"
    mot = get_motivation_file_names()

    for i in mot:
        print(i)
