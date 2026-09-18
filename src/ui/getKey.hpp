#ifndef PSUM_UI_GETKEY_HPP
#define PSUM_UI_GETKEY_HPP

#include <unordered_set>
#include <termios.h>
#include <stdio.h>

namespace psum{

namespace ui {

    enum class Key {
        Tab = 9,
        Enter = 13,
        Space = 32,
        key0 = '0',
        key1 = '1',
        key2 = '2',
        key3 = '3',
        key4 = '4',
        key5 = '5',
        key6 = '6',
        key7 = '7',
        key8 = '8',
        key9 = '9',
        A = 'A',
        B = 'B',
        C = 'C',
        D = 'D',
        E = 'E',
        F = 'F',
        G = 'G',
        H = 'H',
        I = 'I',
        J = 'J',
        K = 'K',
        L = 'L',
        M = 'M',
        N = 'N',
        O = 'O',
        P = 'P',
        Q = 'Q',
        R = 'R',
        S = 'S',
        T = 'T',
        U = 'U',
        V = 'V',
        W = 'W',
        X = 'X',
        Y = 'Y',
        Z = 'Z',
        UpArrow,
        DownArrow,
        LeftArrow,
        RightArrow,
        Plus,         // "+" key (on main keyboard and numpad)
        Minus,        // "-" key
        Asterisk,     // "*" key (numpad)
        Slash,        // "/" key (numpad)
        Period,       // "." key (main keyboard and numpad)
        Comma,        // "," key
        Semicolon,    // ";" key
        Colon,        // ":" key
        Quote,        // "'" key
        Question,     // "?" key
        Exclamation,  // "!" key
        At,           // "@" key
        LeftBracket,  // "[" key
        RightBracket, // "]" key
        Backslash,    // "\" key
        Caret,        // "^" key
        Equal,        // "=" key
        Tilde,        // "~" key
        LeftParen,    // "(" key
        RightParen,   // ")" key
        LeftBrace,    // "{" key
        RightBrace,   // "}" key
        LeftArrowNumPad,
        RightArrowNumPad,
        UpArrowNumPad,
        DownArrowNumPad,
        Unknown
    };

    class KeyboardInput {
    public:
        // Store the valid keys in a static set
        inline static const std::unordered_set<Key>& valid_keys()
        {
            static const std::unordered_set<Key> keys = {
                Key::A, Key::B, Key::C, Key::D, Key::E, Key::F, Key::G, Key::H, Key::I,
                Key::J, Key::K, Key::L, Key::M, Key::N, Key::O, Key::P, Key::Q, Key::R,
                Key::S, Key::T, Key::U, Key::V, Key::W, Key::X, Key::Y, Key::Z, 
                Key::key0, Key::key1, Key::key2, Key::key3, Key::key4, 
                Key::key5, Key::key6, Key::key7, Key::key8, Key::key9,
                Key::Space, Key::Tab, Key::Enter,
                Key::UpArrow, Key::DownArrow, Key::LeftArrow, Key::RightArrow, 
                Key::Plus, Key::Minus, Key::Asterisk, Key::Slash, Key::Period, Key::Comma, Key::Semicolon, Key::Colon,
                Key::Quote, Key::Question, Key::Exclamation, Key::At, Key::LeftBracket,
                Key::RightBracket, Key::Backslash, Key::Caret, Key::Equal, Key::Tilde,
                Key::LeftParen, Key::RightParen, Key::LeftBrace, Key::RightBrace, Key::LeftArrowNumPad,
                Key::RightArrowNumPad, Key::UpArrowNumPad, Key::DownArrowNumPad
            };
            return keys;
        }

        inline static int custom_getch(void) {
        	struct termios tm, tm_old;
        	int fd = 0, ch;
        
        	if (tcgetattr(fd, &tm) < 0) { // save current terminal settings
        		return -1;
        	}
        
        	tm_old = tm;
        	cfmakeraw(&tm); // switch the terminal to raw mode: input is processed byte by byte
        	if (tcsetattr(fd, TCSANOW, &tm) < 0) { // apply the new settings
        		return -1;
        	}
        
        	ch = getchar();
        	if (tcsetattr(fd, TCSANOW, &tm_old) < 0) { // restore the original settings
        		return -1;
        	}
        
        	return ch;
        }

        inline static Key get_key()
        {
            char key = custom_getch(); // Get the pressed key
            if (key == 27)
            { // Escape key, handle arrow keys or function keys
                key = custom_getch();
                if (key == 91)
                { // ESC + '[' sequence for arrow keys
                    key = custom_getch(); // Check for arrow or function keys
                    switch (key)
                    {
                    case 'A': return Key::UpArrow;
                    case 'B': return Key::DownArrow;
                    case 'C': return Key::RightArrow;
                    case 'D': return Key::LeftArrow;
                    default:
                        return Key::Unknown;
                    }
                }
                else return Key::Unknown;
            }
            else if (key >= 'a' && key <= 'z') return static_cast<Key>(key - 'a' + 'A');
            else if (key == '+') return Key::Plus;
            else if (key == '*') return Key::Asterisk;
            else if (key == '-') return Key::Minus;
            else if (key == '/') return Key::Slash;
            else if (key == '.') return Key::Period;
            else if (key == ',') return Key::Comma;
            else if (key == ';') return Key::Semicolon;
            else if (key == ':') return Key::Colon;
            else if (key == '\'') return Key::Quote;
            else if (key == '?') return Key::Question;
            else if (key == '!') return Key::Exclamation;
            else if (key == '@') return Key::At;
            else if (key == '[') return Key::LeftBracket;
            else if (key == ']') return Key::RightBracket;
            else if (key == '\\') return Key::Backslash;
            else if (key == '^') return Key::Caret;
            else if (key == '=') return Key::Equal;
            else if (key == '~') return Key::Tilde;
            else if (key == '(') return Key::LeftParen;
            else if (key == ')') return Key::RightParen;
            else if (key == '{') return Key::LeftBrace;
            else if (key == '}') return Key::RightBrace;
            else if (key >= '0' && key <= '9') return static_cast<Key>(key); // Numeric keys
            else if (valid_keys().contains(static_cast<Key>(key)))  return static_cast<Key>(key); // Return other printable keys
            return Key::Unknown;
        }
    };
}

}


#endif