import curses
import os
from pathlib import Path
import time

from src.config import ConfigManager, ConfigError
from src.curses import CursesMenu, ChatUI

def init_colors(color_config):
    """Initialize color pairs from config."""
    # Map color names to curses color constants
    color_map = {
        "BLACK": curses.COLOR_BLACK,
        "RED": curses.COLOR_RED,
        "GREEN": curses.COLOR_GREEN,
        "YELLOW": curses.COLOR_YELLOW,
        "BLUE": curses.COLOR_BLUE,
        "MAGENTA": curses.COLOR_MAGENTA,
        "CYAN": curses.COLOR_CYAN,
        "WHITE": curses.COLOR_WHITE,
        "DEFAULT": -1
    }

    # Initialize each color pair from config
    curses.init_pair(1, color_map[color_config.menu_selected[0]], color_map[color_config.menu_selected[1]])
    curses.init_pair(2, color_map[color_config.recording_status[0]], color_map[color_config.recording_status[1]])
    curses.init_pair(3, color_map[color_config.ready_status[0]], color_map[color_config.ready_status[1]])
    curses.init_pair(4, color_map[color_config.user_message[0]], color_map[color_config.user_message[1]])
    curses.init_pair(5, color_map[color_config.assistant_message[0]], color_map[color_config.assistant_message[1]])

def show_error(stdscr, message):
    """Display an error message to the user."""
    stdscr.clear()
    stdscr.addstr(0, 0, message)
    stdscr.refresh()
    time.sleep(2)

def main(stdscr):
    # Initialize curses
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)

    # Load configuration
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        config_manager.validate_paths()
    except ConfigError as e:
        print(f"Configuration error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    # Initialize colors from config
    init_colors(config.colors)

    # Create error callback wrapper
    def show_error_callback(message: str):
        show_error(stdscr, message)

    # Create and run the main menu
    main_menu = CursesMenu.create_main_menu(stdscr, config, show_error_callback)
    if main_menu is None:
        return

    selected_settings = main_menu.navigate()

    # Execute the selected action (if any)
    if selected_settings:
        if "Start Chat" in selected_settings:
            main_menu.actions["Start Chat"](stdscr, selected_settings)
        elif "Exit" in selected_settings:
            main_menu.actions["Exit"](stdscr)

if __name__ == "__main__":
    curses.wrapper(main)
