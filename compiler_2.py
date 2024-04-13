# Token types
TOKEN_STRING = 'STRING'
TOKEN_CONCAT = 'CONCAT'
TOKEN_MULTIPLY = 'MULTIPLY'
TOKEN_NEWLINE = 'NEWLINE'
TOKEN_TAB = 'TAB'
TOKEN_EOF = 'EOF'

class Token:
    def __init__(self, type, value=None):
        self.type = type
        self.value = value

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def error(self):
        # Skip the invalid character and continue tokenizing
        self.advance()

    def advance(self):
        self.pos += 1
        if self.pos >= len(self.text):
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def string(self):
        result = ''
        # Skip the opening quotation mark
        self.advance()
        while self.current_char is not None and self.current_char != '"':
            if self.current_char == '\\':
                # Handle escape sequences
                self.advance()
                if self.current_char == 'n':
                    result += '\n'  # Translate '\n' to newline character
                elif self.current_char == 't':
                    result += '\t'  # Translate '\t' to tab character
                else:
                    result += self.current_char  # Handle other escape sequences
            else:
                result += self.current_char
            self.advance()
        # Check if the string is properly terminated with a double quotation mark
        if self.current_char == '"':
            self.advance()  # Skip the closing quotation mark
            return result
        else:
            self.error("Expected '\"' at the end of string")


    def get_next_token(self):
        while self.current_char is not None:

            if self.current_char == '"':
                return Token(TOKEN_STRING, self.string())

            if self.current_char == '+':
                self.advance()
                return Token(TOKEN_CONCAT)

            if self.current_char == '*':
                self.advance()
                return Token(TOKEN_MULTIPLY)
            if self.current_char.isdigit():
                return Token(TOKEN_STRING, self.string())

            if self.current_char == '\n':
                self.advance()
                return Token(TOKEN_NEWLINE)

            if self.current_char == '\t':
                self.advance()
                return Token(TOKEN_TAB)

            self.error()

        return Token(TOKEN_EOF)

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self, message=""):
        print("Error:", message)
        return  # Exit gracefully without raising an exception

    def parse(self):
        return self.expr()
    
    def expr(self):
        result = ""
        while self.current_token.type != TOKEN_EOF:
            # print("Current token type:", self.current_token.type)
            # print("Current token value:", self.current_token.value)
            if self.current_token.type == TOKEN_STRING:
                result += self.current_token.value
                self.eat(TOKEN_STRING)
            elif self.current_token.type == TOKEN_CONCAT:
                self.eat(TOKEN_CONCAT)  # Advance to the next token
            elif self.current_token.type == TOKEN_MULTIPLY:
                self.eat(TOKEN_MULTIPLY)
                # print("Token after '*':", self.current_token.type, self.current_token.value)
                # Check if the next token is a number
                if self.current_token.type != TOKEN_STRING or self.current_token.value is None:
                    self.error("Expected a number after '*'")
                    return ""
                # Repeat the previous string 'result' the specified number of times
                result *= int(self.current_token.value)
                self.eat(TOKEN_STRING)  # Consume the number token
            elif self.current_token.type == TOKEN_NEWLINE:
                self.eat(TOKEN_NEWLINE)
                result += "\n"
            elif self.current_token.type == TOKEN_TAB:
                self.eat(TOKEN_TAB)
                result += "\t"
            else:
                self.error()
        return result

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

class Interpreter:
    def __init__(self, parser):
        self.parser = parser

    def interpret(self):
        return self.parser.parse()


def main():
    print("Welcome to the Custom String Compiler!")
    print("Enter your string expressions using the following syntax:")
    print(" - Use double quotes (\") to enclose strings.")
    print(" - Use '+' to concatenate strings.")
    print(" - Use '*' to multiply a string.")
    print(" - Use '\\n' for newline and '\\t' for tab.")
    print("Example: \"Hello\" + \" \" + \"world!\" * \"3\" ")
    print("Enter 'exit' to quit.")

    while True:
        try:
            text = input(">>> ")
            if text.lower() == 'exit':
                break
        except EOFError:
            break
        if not text:
            continue
        lexer = Lexer(text)
        parser = Parser(lexer)
        interpreter = Interpreter(parser)
        result = interpreter.interpret()
        print(result)

if __name__ == "__main__":
    main()

# example 
# "Hello World ?" + "\n" * "3"