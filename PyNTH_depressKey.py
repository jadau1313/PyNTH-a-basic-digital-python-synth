import pygame
import numpy as np

# Initialize Pygame mixer and sound settings
pygame.init()
pygame.mixer.init()

# Sound settings
sample_rate = 44100  # Sample rate in Hz
freq = 440  # Frequency of the A note in Hz

# Generate one second of the A note (440Hz)
t = np.linspace(0, 1, int(sample_rate), False)
waveform = 0.5 * np.sin(2 * np.pi * freq * t)

# add dictionary of notes with multipliers for each note and octave
keymap = {
    pygame.K_q: 0.5,  # A
    pygame.K_w: 0.5297,  # AsharpBflat
    pygame.K_e: 0.5612,  # B
    pygame.K_r: 0.5946,  # C
    pygame.K_t: 0.6299,  # CsharpDflat
    pygame.K_y: 0.6674,  # D
    pygame.K_u: 0.7071,  # DsharpEflat
    pygame.K_i: 0.7492,  # E
    pygame.K_o: 0.7937,  # F
    pygame.K_p: 0.8409,  # FsharpGflat
    pygame.K_LEFTBRACKET: 0.8909,  # G
    pygame.K_RIGHTBRACKET: 0.9439, #GsharpAflt
    pygame.K_a: 1.0 # A

}

keylist = [pygame.K_q, pygame.K_w, pygame.K_e, pygame.K_r, pygame.K_t, pygame.K_y, pygame.K_u, pygame.K_i, pygame.K_o, pygame.K_p, pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, pygame.K_a]



# Pygame window setup
screen = pygame.display.set_mode((300, 200))
pygame.display.set_caption("Press buttons to play sounds (start with qwerty)")

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key in keylist:
            eventkey = event.key
            multiplier = keymap[eventkey]
            waveformz = 0.5 * np.sin(2 * np.pi * (multiplier*freq) * t)
            waveformz = np.int16(waveformz * 32767)
            sound = pygame.mixer.Sound(waveformz)
            sound.play(-1)
            print(multiplier)
        elif event.type == pygame.KEYUP:
            sound.stop()
            if event.key == pygame.K_q:
                sound.stop()  # Stop playing sound



# Quit Pygame
pygame.quit()
