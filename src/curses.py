import curses
from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple, Optional, Any
from pathlib import Path
import os
import time
from .chat_ui import start_chat

@dataclass
class UIColors:
    menu_selected: Tuple[str, str]
    recording_status: Tuple[str, str]
    ready_status: Tuple[str, str]
    user_message: Tuple[str, str]
    assistant_message: Tuple[str, str]

class CursesMenu:
    def __init__(self, stdscr, options: List[str], actions: Dict[str, Callable]):
        self.stdscr = stdscr
        self.options = options
        self.actions = actions
        self.current_row = 0
        self.settings = {}  # Store selected settings
        # Enable mouse events
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)

    def get_button_coords(self, text: str, y: int, w: int) -> tuple[int, int, int, int]:
        """Calculate button coordinates for mouse click detection."""
        x = w // 2 - len(text) // 2
        return (y, x, y, x + len(text))

    def is_click_within(self, mouse_y: int, mouse_x: int, coords: tuple[int, int, int, int]) -> bool:
        """Check if mouse click is within button coordinates."""
        y1, x1, y2, x2 = coords
        return y1 <= mouse_y <= y2 and x1 <= mouse_x <= x2

    def display_menu(self):
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        
        # Display title if this is a submenu
        if hasattr(self, 'title'):
            title_y = max(0, h // 2 - len(self.options) // 2 - 2)
            title_x = max(0, w // 2 - len(self.title) // 2)
            try:
                self.stdscr.addstr(title_y, title_x, self.title[:w-1])
                if self.settings.get(self.title):
                    selected = f"Selected: {self.settings[self.title]}"
                    selected_x = max(0, w // 2 - len(selected) // 2)
                    self.stdscr.addstr(title_y + 1, selected_x, selected[:w-1])
            except curses.error:
                pass
        
        self.button_coords = []  # Store button coordinates for mouse detection
        
        for i, option in enumerate(self.options):
            x = max(0, w // 2 - len(option) // 2)
            y = max(0, h // 2 - len(self.options) // 2 + i)
            
            # Store button coordinates
            self.button_coords.append(self.get_button_coords(option, y, w))
            
            # Show selection status
            if (hasattr(self, 'title') and self.settings.get(self.title) == option) or \
               (not hasattr(self, 'title') and option in self.settings):
                option = f"[*] {option}"
            
            try:
                if i == self.current_row:
                    self.stdscr.attron(curses.color_pair(1))
                    self.stdscr.addstr(y, x, option[:w-1])
                    self.stdscr.attroff(curses.color_pair(1))
                else:
                    self.stdscr.addstr(y, x, option[:w-1])
            except curses.error:
                pass
        
        # Add help text at bottom
        try:
            help_text = "↑/↓: Navigate | Enter: Select | ESC: Back | Click: Select"
            help_x = max(0, w//2 - len(help_text)//2)
            self.stdscr.addstr(h-1, help_x, help_text[:w-1])
        except curses.error:
            pass
        
        self.stdscr.refresh()

    def navigate(self):
        while True:
            self.display_menu()
            key = self.stdscr.getch()

            if key == curses.KEY_MOUSE:
                try:
                    _, mouse_x, mouse_y, _, button_state = curses.getmouse()
                    if button_state & curses.BUTTON1_CLICKED:
                        selected = self.handle_mouse_click(mouse_y, mouse_x)
                        if selected:
                            if selected in self.actions:
                                if callable(self.actions[selected]):
                                    submenu_result = self.actions[selected](self.stdscr, self.settings.copy())
                                    if submenu_result:
                                        self.settings.update(submenu_result)
                                else:
                                    self.settings[selected] = True
                            elif hasattr(self, 'title'):
                                return {self.title: selected}
                except curses.error:
                    pass

            elif key == curses.KEY_UP and self.current_row > 0:
                self.current_row -= 1
            elif key == curses.KEY_DOWN and self.current_row < len(self.options) - 1:
                self.current_row += 1
            elif key == curses.KEY_ENTER or key in [10, 13]:
                selected_option = self.options[self.current_row]
                
                if selected_option in self.actions:
                    if callable(self.actions[selected_option]):
                        submenu_result = self.actions[selected_option](self.stdscr, self.settings.copy())
                        if submenu_result:
                            self.settings.update(submenu_result)
                    else:
                        self.settings[selected_option] = True
                elif hasattr(self, 'title'):
                    return {self.title: selected_option}
                
                self.display_menu()
                
            elif key == 27:  # ESC key
                if hasattr(self, 'title'):
                    return self.settings
                break
                
        return self.settings

    def create_submenu(self, title: str, options: List[str], default_value: str = None):
        def submenu(stdscr, settings=None):
            submenu = CursesMenu(stdscr, options, {})
            submenu.title = title
            submenu.settings = self.settings.copy()
            
            if default_value in options:
                submenu.current_row = options.index(default_value)
            if title in self.settings:
                selected = self.settings[title]
                if selected in options:
                    submenu.current_row = options.index(selected)
            
            result = submenu.navigate()
            if result:
                self.settings.update(result)
            return result
        return submenu

    @staticmethod
    def scan_llm_models(llm_path: Path, error_callback: Callable[[str], None]) -> list[str]:
        """Scan for available LLM models."""
        try:
            print(f"Scanning directory: {llm_path} (absolute: {llm_path.absolute()})")  # Debug print
            if not llm_path.exists():
                error_callback(f"Directory not found: {llm_path} (absolute: {llm_path.absolute()})")
                return []
            
            llm_models = []
            for file in os.listdir(llm_path):
                full_path = llm_path / file
                print(f"Found file: {full_path} (exists: {full_path.exists()})")  # Debug print
                if file.endswith(".gguf"):
                    llm_models.append(file)
            
            if not llm_models:
                error_callback(f"No GGUF models found in {llm_path} (absolute: {llm_path.absolute()})")
                return []
            
            print(f"Found models: {llm_models}")  # Debug print
            return llm_models
        except Exception as e:
            error_callback(f"Error scanning LLM models: {str(e)}")
            import traceback
            print(f"Exception traceback: {traceback.format_exc()}")  # Debug print
            return []

    @staticmethod
    def load_voices(voices_path: Path, show_error_callback) -> list[str]:
        """Load available TTS voices."""
        try:
            voices = {}
            with open(voices_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        voice_name = parts[1].strip()
                        voices[voice_name] = voice_name
            if not voices:
                show_error_callback("No voices found in voices.txt")
                return []
            voice_names = list(voices.values())
            voice_names.append("[ Blend Voices ]")
            return voice_names
        except FileNotFoundError:
            show_error_callback("Error: voices.txt not found")
            return []
        except Exception as e:
            show_error_callback(f"Error loading voices: {str(e)}")
            return []

    @staticmethod
    def scan_prompt_files(prompts_dir: Path) -> list[str]:
        """Scan for available prompt XML files."""
        if not prompts_dir.exists():
            return ["template.xml"]  # Default
        return [f.name for f in prompts_dir.glob("*.xml")]

    @classmethod
    def create_main_menu(cls, stdscr, config, error_callback: Callable[[str], None]):
        """Create and initialize the main menu with all submenus."""
        # Scan for resources
        llm_path = Path(config.llm.model_path)
        print(f"LLM path from config: {llm_path}")  # Debug print
        if llm_path.is_file():  # If it's a file path, get the parent directory
            llm_path = llm_path.parent
        print(f"Using LLM directory: {llm_path} (absolute: {llm_path.absolute()})")  # Debug print
        
        llm_models = cls.scan_llm_models(llm_path, error_callback)
        if not llm_models:
            return None

        voice_names = cls.load_voices(Path(config.tts.voices_path), error_callback)
        if not voice_names:
            return None

        prompt_files = cls.scan_prompt_files(Path(config.system_prompt_path).parent)
        if not prompt_files:
            error_callback("No prompt files found in prompts directory")
            return None

        # Create menu options
        menu_options = ["LLM Model", "TTS Voice", "System Prompt", "Start Chat", "Exit"]
        menu = cls(stdscr, menu_options, {})

        # Create submenus
        llm_submenu = menu.create_submenu("LLM Model", llm_models)
        tts_submenu = menu.create_submenu("TTS Voice", voice_names)
        prompt_submenu = menu.create_submenu("System Prompt", prompt_files)

        def handle_voice_blend(stdscr, settings):
            if settings.get("TTS Voice") == "[ Blend Voices ]":
                voice1_submenu = menu.create_submenu("Voice 1", voice_names[:-1])
                voice2_submenu = menu.create_submenu("Voice 2", voice_names[:-1])
                ratio_options = [str(i) for i in range(0, 101, 10)]
                ratio_submenu = menu.create_submenu("Blend Ratio", ratio_options)

                voice1_result = voice1_submenu(stdscr)
                if not voice1_result or "Voice 1" not in voice1_result:
                    return None

                voice2_result = voice2_submenu(stdscr)
                if not voice2_result or "Voice 2" not in voice2_result:
                    return None

                ratio_result = ratio_submenu(stdscr)
                if not ratio_result or "Blend Ratio" not in ratio_result:
                    return None

                settings["Voice 1"] = voice1_result["Voice 1"]
                settings["Voice 2"] = voice2_result["Voice 2"]
                settings["Blend Ratio"] = int(ratio_result["Blend Ratio"])
            return settings

        # Define menu actions
        menu.actions = {
            "LLM Model": llm_submenu,
            "TTS Voice": lambda stdscr, settings: handle_voice_blend(stdscr, tts_submenu(stdscr, settings) or {}),
            "System Prompt": prompt_submenu,
            "Start Chat": lambda stdscr, settings: start_chat(stdscr, config, settings) 
                if all(key in settings for key in ["LLM Model", "TTS Voice", "System Prompt"]) 
                else error_callback("Please select Model, Voice, and Prompt."),
            "Exit": lambda stdscr: curses.endwin(),
        }

        return menu

class ChatUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.header_win = None
        self.chat_win = None
        self.status_win = None
        self.button_positions = {}
        self.create_windows()

    def create_windows(self):
        """Create and initialize the main chat windows."""
        h, w = self.stdscr.getmaxyx()
        
        # Create header window (3 lines for buttons)
        self.header_win = curses.newwin(3, w, 0, 0)
        self.header_win.box()
        
        # Create chat history window (main area)
        self.chat_win = curses.newwin(h - 5, w, 3, 0)
        self.chat_win.box()
        self.chat_win.scrollok(True)
        
        # Create status bar window (bottom line)
        self.status_win = curses.newwin(2, w, h - 2, 0)
        self.status_win.box()

    def update_header(self, recording: bool = False, is_ptt_mode: bool = True) -> Dict[str, Tuple[int, int, int]]:
        """Update the header window with current status and buttons."""
        h, w = self.header_win.getmaxyx()
        self.header_win.clear()
        self.header_win.box()
        
        title = "Voice Chat Interface"
        self.header_win.addstr(0, (w - len(title)) // 2, title)
        
        # Add buttons
        buttons = [
            "[ Clear Chat ]",
            "[ Restart ]",
            "[ Settings ]",
            "[ Redo Last ]",
            "[ Stop ]",
            f"[ {'PTT' if is_ptt_mode else 'Passive'} Mode ]"
        ]
        
        # Calculate button positions
        x_pos = 2
        button_positions = {}
        for i, btn in enumerate(buttons):
            button_positions[btn.strip('[]').strip().lower()] = (1, x_pos, len(btn))
            self.header_win.addstr(1, x_pos, btn)
            x_pos += len(btn) + 2
        
        # Status indicator
        if recording:
            status = "● Recording..."
            self.header_win.addstr(1, w - len(status) - 2, status, curses.color_pair(2))
        else:
            status = "◯ Ready"
            self.header_win.addstr(1, w - len(status) - 2, status, curses.color_pair(3))
        
        # Controls help text
        controls = "SPACE: Record/Stop | ↑/↓: Scroll | ESC: Exit | Mouse: Click buttons"
        if not is_ptt_mode:
            controls = "Auto Recording | ↑/↓: Scroll | ESC: Exit | Mouse: Click buttons"
        self.header_win.addstr(2, (w - len(controls)) // 2, controls)
        
        self.header_win.refresh()
        self.button_positions = button_positions
        return button_positions

    def update_status(self, message: str = ""):
        """Update the status bar with a message."""
        h, w = self.status_win.getmaxyx()
        self.status_win.clear()
        self.status_win.box()
        if message:
            # Truncate message if too long
            if len(message) > w - 4:
                message = message[:w-7] + "..."
            self.status_win.addstr(0, 2, message)
        self.status_win.refresh()

    def display_chat(self, chat_history: List[str], scroll_offset: int = 0):
        """Display chat history in the chat window with proper formatting."""
        h, w = self.chat_win.getmaxyx()
        self.chat_win.clear()
        self.chat_win.box()
        
        # Calculate available space for messages
        max_lines = h - 2  # Account for box borders
        available_width = w - 4  # Account for box borders and padding
        
        # Format and wrap messages
        formatted_lines = []
        for message in chat_history:
            # Split into speaker and content
            if message.startswith("You: "):
                speaker = "You"
                content = message[4:]
                prefix = "→ "
                color = curses.color_pair(4)  # User message color
            else:
                speaker = "Assistant"
                content = message[11:]  # Remove "Assistant: "
                prefix = "← "
                color = curses.color_pair(5)  # Assistant message color
            
            # Wrap message content
            wrapped_content = []
            current_line = ""
            words = content.split()
            
            for word in words:
                if len(current_line) + len(word) + 1 <= available_width - len(prefix) - len(speaker) - 2:
                    current_line += (word + " ")
                else:
                    if current_line:
                        wrapped_content.append(current_line.strip())
                    current_line = word + " "
            if current_line:
                wrapped_content.append(current_line.strip())
            
            # Add formatted lines
            if wrapped_content:
                # First line with speaker
                formatted_lines.append((f"{prefix}{speaker}: {wrapped_content[0]}", color))
                # Continuation lines
                for line in wrapped_content[1:]:
                    formatted_lines.append((f"{' ' * (len(prefix) + len(speaker) + 2)}{line}", color))
                # Add spacing between messages
                formatted_lines.append(("", None))
        
        # Display messages with scrolling
        start_idx = max(0, len(formatted_lines) - max_lines - scroll_offset)
        display_lines = formatted_lines[start_idx:start_idx + max_lines]
        
        for i, (line, color) in enumerate(display_lines):
            if i < max_lines:
                if color:
                    self.chat_win.attron(color)
                self.chat_win.addstr(i + 1, 2, line[:available_width])
                if color:
                    self.chat_win.attroff(color)
        
        self.chat_win.refresh()

    def handle_button_click(self, mouse_y: int, mouse_x: int) -> Optional[str]:
        """Handle button clicks in the header."""
        for btn, (y, x, width) in self.button_positions.items():
            if mouse_y == y and x <= mouse_x < x + width:
                return btn
        return None 