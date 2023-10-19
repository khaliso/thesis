import psutil

def get_graphics_cards():
    graphics_cards = []
    for device in psutil.iter_devicesshow():
        if device.device_type() == psutil._psutil_linux.device_type.DEV_TYPE_GPU:
            graphics_cards.append(device.name())
    return graphics_cards

# Get the list of available graphics cards
available_graphics_cards = get_graphics_cards()

# Print the list of available graphics cards
for card in available_graphics_cards:
    print(card)
