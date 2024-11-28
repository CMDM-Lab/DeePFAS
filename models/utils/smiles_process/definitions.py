"""
Define Some maps, lists of token.

Author: Heng Wang
Date: 1/23/2024
"""

from types import MappingProxyType

INITIAL_CHAR = '<SOS>'
FINAL_CHAR = '<EOS>'
PAD_CHAR = '<PAD>'
UNKNOWN_CAHR = '<UNK>'
VOC = (  # remove i, @ , :
    '#',
    '%',
    '(',
    ')',
    '+',
    '-',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '=',
    'B',
    'Br',
    'C',
    'Cl',
    'F',
    'H',
    'I',
    'N',
    'O',
    'P',
    'S',
    '[',
    ']',
    'c',
    'n',
    'o',
    'p',
    's',
    INITIAL_CHAR,
    FINAL_CHAR,
    PAD_CHAR,
    UNKNOWN_CAHR
)

ELEMENTS = MappingProxyType({'C': 1, 'N': 1, 'H': 1, 'O': 1, 'P': 1, 'S': 1, 'Cl': 1, 'Br': 1, 'F': 1, 'I': 1, 'B': 1})
VOC_MAP = MappingProxyType({token: idx for idx, token in enumerate(VOC)})
