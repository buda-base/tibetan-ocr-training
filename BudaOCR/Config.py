CHARSET = "!#%'°^$`()+-./0123456789:=?@ADHIMNRSTUWXY[\]abcdefghijklmnoprstuwyxz|~§&äüö<>;卍卐ऽ}"
STACK_FILE = "tib-stacks_v2.txt"

# Tibetan character ranges
CONSONANTS = '\u0F40-\u0F6C'  # ཀ-ཌ, ཎ-ཱ, ི-ུ, ཱུ-ཹ, ེ-ཻ, ོ-ཽ, ཾ-ཿ
VOWELS = '\u0F71-\u0F7E'  # ཱ, ི, ུ, ཱུ, ྲྀ, ཷ, ླྀ, ཹ, ེ, ཻ, ོ, ཽ, ཾ, ཿ
SUBSCRIPTS = '\u0F90-\u0FB8'  # ྐ-ྸ, ླ-ྼ, ྾-྿

NUMBERS = '\u0F20-\u0F33'  # ༠-༳

# Standalone characters broken into ranges and named
MARKS_INITIAL = '\u0F04-\u0F12'  # ༄-༒
MARKS_CLOSING = '\u0F13-\u0F17'  # ༓-༗
MARKS_SECONDARY = '\u0F1A-\u0F34'  # ༚-༴
MARKS_ENCLOSING = '\u0F3A-\u0F3D'  # ༺-༽
MARKS_EXTRA = '\u0FBE-\u0FDA'  # ༾-༚

# Combine all standalone symbols
STANDALONE_SYMBOLS = f'{MARKS_INITIAL}{MARKS_CLOSING}{MARKS_SECONDARY}{MARKS_ENCLOSING}{MARKS_EXTRA}{NUMBERS}'

# Special symbols
SYLLABLES_AND_MARKS = '\u0F00-\u0F03'  # ༀ-༃
MARKS_CARET_AND_OTHERS = '\u0F35-\u0F39'  # ༵-༹
SIGN_RNAM_BCAD = '\u0F7F-\u0F80'  # ཿ-ྀ
MARKS_PALUTA_AND_OTHERS = '\u0F85-\u0F8B'  # ྅-ྋ
SHAD_CHARACTERS = '\u0F08\u0F0C\u0F0D\u0F0E\u0F0F\u0F11\u0F14\u0F15\u0F16\u0F17'  # ༈, ༌, །, ༎, ༏, ༐, ༑, ༒, ༓, ༔, ༕, ༖, ༗

# Combine all special symbols and shad characters
SPECIAL_SYMBOLS = f'{SYLLABLES_AND_MARKS}{MARKS_CARET_AND_OTHERS}{SIGN_RNAM_BCAD}{MARKS_PALUTA_AND_OTHERS}{SHAD_CHARACTERS}'

VALIDATION_PATTERN = rf'''
^
(
    (
        [{CONSONANTS}]([{VOWELS}{SUBSCRIPTS}]*)  # Valid consonant followed by vowels or subscripts
    )|
    (
        [{STANDALONE_SYMBOLS}{NUMBERS}{SPECIAL_SYMBOLS}]                    # Standalone characters, numbers, or special symbols alone
    )
)*
'''
