import pygame

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Distance Progress Bar")

# Colors
WHITE = (255, 255, 255)
BLUE = (50, 150, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Progress bar settings
BAR_Y = HEIGHT // 2
BAR_HEIGHT = 40
MAX_DISTANCE = 300  # Define max distance for full bar
CENTER_X = WIDTH // 2

running = True
while running:
    screen.fill(WHITE)

    # Simulated hand distance (Replace with real data from ZED)
    left_hand_x = 200  # Example left hand X position
    right_hand_x = 600  # Example right hand X position
    hand_distance = abs(right_hand_x - left_hand_x)

    # Map hand distance to progress bar width
    progress_width = min(hand_distance, MAX_DISTANCE)  # Cap at max distance

    # Draw the progress bar (expanding from center)
    pygame.draw.rect(screen, BLUE, (CENTER_X - progress_width // 2, BAR_Y, progress_width, BAR_HEIGHT))

    # Check if limit is reached
    if hand_distance >= MAX_DISTANCE:
        font = pygame.font.Font(None, 36)
        text = font.render("Limit Reached!", True, RED)
        screen.blit(text, (CENTER_X - 50, BAR_Y - 50))  # Display above the bar

    # Event handling (quit event)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

pygame.quit()
