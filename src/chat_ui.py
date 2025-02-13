"""Chat UI functionality module."""
import curses
import time
from .chat import ChatManager

def start_chat(stdscr, config, settings):
    """Initialize and run the chat interface."""
    from .curses import ChatUI  # Import here to avoid circular imports
    
    # Initialize UI
    chat_ui = ChatUI(stdscr)
    
    # Create chat manager
    chat_manager = ChatManager(config, settings, chat_ui.update_status)
    
    # Set up chat update callback
    chat_manager.on_chat_updated = lambda: chat_ui.display_chat(chat_manager.state.chat_history, chat_manager.state.scroll_offset)
    
    # Initialize models
    if not chat_manager.initialize_models():
        return
    
    # Initial UI update
    button_positions = chat_ui.update_header(is_ptt_mode=chat_manager.state.is_ptt_mode)
    chat_ui.update_status("Ready to chat! Press SPACE to start recording.")
    chat_ui.display_chat(chat_manager.state.chat_history)

    while True:
        key = stdscr.getch()
        
        if key == curses.KEY_MOUSE:
            try:
                _, mouse_x, mouse_y, _, button_state = curses.getmouse()
                if button_state & curses.BUTTON1_CLICKED:
                    clicked = chat_ui.handle_button_click(mouse_y, mouse_x)
                    if clicked == 'clear':
                        chat_manager.clear_history()
                        chat_ui.display_chat(chat_manager.state.chat_history)
                    elif clicked == 'restart':
                        chat_ui.update_status("Restarting models...")
                        if chat_manager.initialize_models():
                            chat_manager.clear_history()
                            chat_ui.display_chat(chat_manager.state.chat_history)
                            chat_ui.update_status("System restarted successfully!")
                    elif clicked == 'settings':
                        return
                    elif clicked == 'mode':
                        chat_manager.toggle_ptt_mode()
                        button_positions = chat_ui.update_header(is_ptt_mode=chat_manager.state.is_ptt_mode)
                        if not chat_manager.state.is_ptt_mode:
                            stdscr.nodelay(1)  # Set non-blocking mode
                        else:
                            stdscr.nodelay(0)  # Set blocking mode
                    elif clicked == 'redo':
                        if chat_manager.redo_last_message():
                            chat_ui.display_chat(chat_manager.state.chat_history)
                    elif clicked == 'stop':
                        chat_manager.stop_processing()
            except curses.error:
                pass
        
        elif key == ord(' ') and chat_manager.state.is_ptt_mode:
            if chat_manager.state.recording_stream is None:
                if chat_manager.start_recording():
                    button_positions = chat_ui.update_header(recording=True, is_ptt_mode=True)
                    chat_ui.update_status("Recording... Press SPACE again to stop.")
            else:
                result = chat_manager.stop_recording()
                button_positions = chat_ui.update_header(recording=False, is_ptt_mode=True)
                
                if result:
                    audio_data, sample_rate = result
                    chat_ui.update_status("Transcribing speech...")
                    user_message = chat_manager.transcribe(audio_data, sample_rate)
                    if chat_manager.process_voice_input(user_message):
                        chat_ui.display_chat(chat_manager.state.chat_history)
                else:
                    chat_ui.update_status("No audio recorded. Try again.")
                    time.sleep(2)
                
                chat_ui.update_status("Ready to chat!")
        
        elif not chat_manager.state.is_ptt_mode and not chat_manager.state.processing:
            user_message = chat_manager.handle_passive_recording()
            if chat_manager.process_voice_input(user_message):
                chat_ui.display_chat(chat_manager.state.chat_history)
                time.sleep(1)  # Brief pause before starting next recording
        
        elif key == 27:  # ESC key
            chat_manager.cleanup()
            stdscr.nodelay(0)  # Reset to blocking mode
            break
        
        elif key == curses.KEY_UP:
            if chat_manager.state.scroll_offset < len(chat_manager.state.chat_history) * 2:
                chat_manager.state.scroll_offset += 1
                chat_ui.display_chat(chat_manager.state.chat_history, chat_manager.state.scroll_offset)
        
        elif key == curses.KEY_DOWN:
            if chat_manager.state.scroll_offset > 0:
                chat_manager.state.scroll_offset -= 1
                chat_ui.display_chat(chat_manager.state.chat_history, chat_manager.state.scroll_offset) 