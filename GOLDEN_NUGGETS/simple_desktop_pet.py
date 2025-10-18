#!/usr/bin/env python3
"""
Simple Desktop Pet with Proper Click-Through Functionality
Based on proven techniques from Desktop Goose and Shijima-Qt

This implementation works by:
1. Creating a transparent, always-on-top window
2. Using proper Windows API calls for click-through
3. Keeping the character interactive while making background transparent
4. Proper context menu and chat functionality
"""

import sys
import json
import random
import math
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QMenu, QSystemTrayIcon, 
                            QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, 
                            QPushButton, QMessageBox)
from PyQt6.QtCore import QTimer, Qt, QPoint, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QPixmap, QIcon, QAction, QPainter, QColor, QFont, QPalette

try:
    # Windows-specific imports for proper click-through
    import ctypes
    from ctypes import wintypes
    import win32api
    import win32con
    import win32gui
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("Win32 extensions not available. Click-through will not work properly on Windows.")

class ChatWindow(QDialog):
    """Simple chat window that appears near the character"""
    
    def __init__(self, parent=None, position=None):
        super().__init__(parent)
        self.setWindowTitle("Chat with Pet")
        self.setFixedSize(400, 500)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        
        if position:
            self.move(position.x() + 100, position.y() - 50)
        
        self.setup_ui()
        self.setup_style()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.append("🐾 Hi! I'm your desktop pet. What would you like to talk about?")
        layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.returnPressed.connect(self.send_message)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        self.setLayout(layout)
        
    def setup_style(self):
        """Apply dark theme styling"""
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 rgba(30,30,40,0.95), stop:1 rgba(40,40,50,0.95));
                border: 1px solid rgba(100,100,150,0.3);
                border-radius: 12px;
            }
            QTextEdit {
                background: rgba(20,20,30,0.7);
                color: white;
                border: 1px solid rgba(100,100,150,0.2);
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }
            QLineEdit {
                background: rgba(20,20,30,0.5);
                color: white;
                border: 1px solid rgba(100,100,150,0.3);
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #4a90e2, stop:1 #6a5acd);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #5a9ff2, stop:1 #7a6add);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #3a80d2, stop:1 #5a4abd);
            }
        """)
    
    def send_message(self):
        message = self.input_field.text().strip()
        if not message:
            return
            
        self.input_field.clear()
        self.chat_display.append(f"You: {message}")
        
        # Simple responses
        responses = [
            "That's interesting! Tell me more 🐾",
            "I love chatting with you! 😊",
            "Woof! That sounds exciting! 🎾",
            "I'm just a simple pet, but I'm happy to listen! 🐕",
            "Thanks for talking to me! It makes me happy! ❤️"
        ]
        response = random.choice(responses)
        self.chat_display.append(f"Pet: {response}")
        
        # Scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

class DesktopPet(QWidget):
    """Main desktop pet widget with proper click-through functionality"""
    
    def __init__(self):
        super().__init__()
        self.dragging = False
        self.drag_position = QPoint()
        self.animation_timer = QTimer()
        self.behavior_timer = QTimer()
        self.click_through_enabled = True
        self.chat_window = None
        self.animation_offset = 0
        self.idle_time = 0
        self.expressions = ["🐾", "😊", "🎾", "❤️", "😴", "🤔", "🎭", "🌟"]
        self.current_expression = 0
        
        self.setup_window()
        self.setup_ui()
        self.setup_animations()
        self.setup_tray()
        self.apply_click_through()
        
    def setup_window(self):
        """Configure the main pet window"""
        # Remove window frame and make it stay on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        
        # Make background transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        
        # Set window size
        self.setFixedSize(120, 120)
        
        # Position in bottom right corner initially
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - 140, screen.height() - 140)
        
    def setup_ui(self):
        """Setup the UI elements"""
        self.pet_label = QLabel(self)
        self.pet_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pet_label.setStyleSheet("""
            QLabel {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.8,
                    fx:0.5, fy:0.5, stop:0 rgba(255,255,255,0.1), 
                    stop:1 rgba(255,255,255,0.0));
                border-radius: 60px;
                font-size: 48px;
                color: white;
                padding: 10px;
            }
        """)
        self.pet_label.setText(self.expressions[self.current_expression])
        self.pet_label.setGeometry(0, 0, 120, 120)
        
        # Make only the pet label interactive
        self.pet_label.mousePressEvent = self.mousePressEvent
        self.pet_label.mouseMoveEvent = self.mouseMoveEvent
        self.pet_label.mouseReleaseEvent = self.mouseReleaseEvent
        self.pet_label.contextMenuEvent = self.contextMenuEvent
        
    def setup_animations(self):
        """Setup animation timers"""
        # Bobbing animation
        self.animation_timer.timeout.connect(self.animate_bob)
        self.animation_timer.start(50)  # 20fps
        
        # Behavior changes
        self.behavior_timer.timeout.connect(self.update_behavior)
        self.behavior_timer.start(2000)  # Every 2 seconds
        
    def setup_tray(self):
        """Setup system tray icon and menu"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
            
        self.tray_icon = QSystemTrayIcon(self)
        
        # Create tray menu
        tray_menu = QMenu()
        
        open_chat_action = QAction("💬 Open Chat", self)
        open_chat_action.triggered.connect(self.open_chat)
        
        toggle_click_action = QAction("🖱️ Toggle Click-Through", self)
        toggle_click_action.triggered.connect(self.toggle_click_through)
        
        quit_action = QAction("❌ Quit", self)
        quit_action.triggered.connect(QApplication.quit)
        
        tray_menu.addAction(open_chat_action)
        tray_menu.addAction(toggle_click_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        
        # Set tray icon (use the current expression)
        self.update_tray_icon()
        self.tray_icon.show()
        
    def update_tray_icon(self):
        """Update the tray icon with current expression"""
        if hasattr(self, 'tray_icon'):
            # Create a simple pixmap for the tray icon
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(pixmap)
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 20))
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, 
                           self.expressions[self.current_expression])
            painter.end()
            
            self.tray_icon.setIcon(QIcon(pixmap))
    
    def apply_click_through(self):
        """Apply Windows-specific click-through functionality"""
        if not HAS_WIN32 or not self.click_through_enabled:
            return
            
        try:
            hwnd = int(self.winId())
            
            # Get current window style
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            
            # Add transparent and layered style
            style |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
            
            # Apply the new style
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)
            
            # Set the window region to only include the pet area
            # This makes only the character clickable, everything else passes through
            region_size = 80  # Smaller than the full window
            offset = (120 - region_size) // 2
            
            region = win32gui.CreateEllipticRgn(
                offset, offset, 
                offset + region_size, offset + region_size
            )
            win32gui.SetWindowRgn(hwnd, region, True)
            
        except Exception as e:
            print(f"Failed to apply click-through: {e}")
    
    def remove_click_through(self):
        """Remove click-through functionality"""
        if not HAS_WIN32:
            return
            
        try:
            hwnd = int(self.winId())
            
            # Get current window style
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            
            # Remove transparent style
            style &= ~win32con.WS_EX_TRANSPARENT
            
            # Apply the new style
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)
            
            # Remove window region (make entire window clickable)
            win32gui.SetWindowRgn(hwnd, None, True)
            
        except Exception as e:
            print(f"Failed to remove click-through: {e}")
    
    def toggle_click_through(self):
        """Toggle click-through functionality"""
        self.click_through_enabled = not self.click_through_enabled
        
        if self.click_through_enabled:
            self.apply_click_through()
            print("Click-through enabled")
        else:
            self.remove_click_through()
            print("Click-through disabled")
    
    def animate_bob(self):
        """Simple bobbing animation"""
        if not self.dragging:
            self.animation_offset += 0.2
            bob_offset = int(math.sin(self.animation_offset) * 5)
            
            # Apply the bobbing offset to the label
            current_pos = self.pet_label.pos()
            self.pet_label.move(current_pos.x(), 10 + bob_offset)
    
    def update_behavior(self):
        """Update pet behavior and expression"""
        self.idle_time += 1
        
        # Change expression occasionally
        if random.random() < 0.3:
            self.current_expression = random.randint(0, len(self.expressions) - 1)
            self.pet_label.setText(self.expressions[self.current_expression])
            self.update_tray_icon()
    
    def mousePressEvent(self, event):
        """Handle mouse press for dragging"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            
            # React to being clicked
            self.current_expression = 1  # Happy face
            self.pet_label.setText(self.expressions[self.current_expression])
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging"""
        if event.buttons() == Qt.MouseButton.LeftButton and self.dragging:
            new_pos = event.globalPosition().toPoint() - self.drag_position
            
            # Keep within screen bounds
            screen = QApplication.primaryScreen().availableGeometry()
            new_pos.setX(max(0, min(new_pos.x(), screen.width() - self.width())))
            new_pos.setY(max(0, min(new_pos.y(), screen.height() - self.height())))
            
            self.move(new_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            event.accept()
    
    def contextMenuEvent(self, event):
        """Show context menu on right click"""
        context_menu = QMenu(self)
        context_menu.setStyleSheet("""
            QMenu {
                background: rgba(30,30,40,0.95);
                border: 1px solid rgba(100,100,150,0.3);
                border-radius: 8px;
                color: white;
                font-size: 13px;
            }
            QMenu::item {
                padding: 8px 16px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: rgba(100,100,150,0.3);
            }
        """)
        
        open_chat_action = QAction("💬 Open Chat", self)
        open_chat_action.triggered.connect(self.open_chat)
        
        dance_action = QAction("💃 Dance", self)
        dance_action.triggered.connect(self.dance)
        
        sleep_action = QAction("😴 Sleep", self)
        sleep_action.triggered.connect(self.sleep)
        
        toggle_click_action = QAction(
            "🔒 Disable Click-Through" if self.click_through_enabled else "🔓 Enable Click-Through", 
            self
        )
        toggle_click_action.triggered.connect(self.toggle_click_through)
        
        context_menu.addAction(open_chat_action)
        context_menu.addAction(dance_action)
        context_menu.addAction(sleep_action)
        context_menu.addSeparator()
        context_menu.addAction(toggle_click_action)
        
        context_menu.exec(event.globalPos())
    
    def open_chat(self):
        """Open the chat window"""
        if self.chat_window is None or not self.chat_window.isVisible():
            self.chat_window = ChatWindow(self, self.pos())
            self.chat_window.show()
        else:
            self.chat_window.raise_()
            self.chat_window.activateWindow()
    
    def dance(self):
        """Make the pet dance"""
        self.current_expression = 6  # Dance emoji
        self.pet_label.setText(self.expressions[self.current_expression])
        
        # Simple dance animation
        original_pos = self.pos()
        for i in range(4):
            QTimer.singleShot(i * 200, lambda: self.move(
                original_pos.x() + random.randint(-10, 10),
                original_pos.y() + random.randint(-10, 10)
            ))
        
        QTimer.singleShot(800, lambda: self.move(original_pos))
        QTimer.singleShot(1000, lambda: self.pet_label.setText(self.expressions[0]))
    
    def sleep(self):
        """Make the pet sleep"""
        self.current_expression = 4  # Sleep emoji
        self.pet_label.setText(self.expressions[self.current_expression])
        self.update_tray_icon()
    
    def closeEvent(self, event):
        """Handle close event"""
        # Hide to tray instead of closing
        self.hide()
        if hasattr(self, 'tray_icon'):
            self.tray_icon.showMessage(
                "Desktop Pet",
                "Pet is now in the system tray. Right-click the tray icon to open the menu.",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
        event.ignore()

def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Keep running when window is closed
    
    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(None, "System Tray",
                           "I couldn't detect a system tray on this system.")
        sys.exit(1)
    
    # Create and show the pet
    pet = DesktopPet()
    pet.show()
    
    # Show startup message
    if hasattr(pet, 'tray_icon'):
        QTimer.singleShot(1000, lambda: pet.tray_icon.showMessage(
            "Desktop Pet Started!",
            "I'm your new desktop companion! 🐾\n\nFeatures:\n• Click-through enabled (clicks pass through empty areas)\n• Drag me around\n• Right-click for menu\n• Chat with me\n• I live in your system tray",
            QSystemTrayIcon.MessageIcon.Information,
            5000
        ))
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
