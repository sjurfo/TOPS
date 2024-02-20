def load():
    return {
        'base_mva': 900,
        'f': 50,
        'slack_bus': 'B3',

        'buses': [
            ['name',    'V_n'],
            ['B1',      20],
            ['B2',      20],
            ['B3',      20],
            ['B4',      20],
            ['B5',      230],
            ['B6',      230],
            ['B7',      230],
            ['B8',      230],
            ['B9',      230],
            ['B10',     230],
            ['B11',     230],
        ],

        'lines': [
            ['name',    'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit',     'R',    'X',    'B'],
            ['L5-6',    'B5',       'B6',       25,         100,    230,    'p.u.',     1e-4,   1e-3,   1.75e-3],
            ['L6-7',    'B6',       'B7',       10,         100,    230,    'p.u.',     1e-4,   1e-3,   1.75e-3],
            ['L7-8-1',  'B7',       'B8',       50,        100,    230,    'p.u.',     1e-4,   1e-3,   1.75e-3],
            ['L7-8-2',  'B7',       'B8',       50,        100,    230,    'p.u.',     1e-4,   1e-3,   1.75e-3],
            ['L8-9-1',  'B8',       'B9',       50,        100,    230,    'p.u.',     1e-4,   1e-3,   1.75e-3],
            ['L8-9-2',  'B8',       'B9',       50,        100,    230,    'p.u.',     1e-4,   1e-3,   1.75e-3],
            ['L9-10',   'B9',       'B10',      10,         100,    230,    'p.u.',     1e-4,   1e-3,   1.75e-3],
            ['L10-11',  'B10',      'B11',      25,         100,    230,    'p.u.',     1e-4,   1e-3,   1.75e-3],
        ],

        'transformers': [
            ['name',    'from_bus', 'to_bus',   'S_n',  'V_n_from', 'V_n_to',   'R',    'X'],
            ['T1',      'B1',       'B5',       900,    20,         230,        0,      0.15],
            ['T2',      'B2',       'B6',       900,    20,         230,        0,      0.15],
            ['T3',      'B3',       'B11',      900,    20,         230,        0,      0.15],
            ['T4',      'B4',       'B10',      900,    20,         230,        0,      0.15],
        ],

        'loads': [
            ['name',    'bus',  'P',    'Q',    'model'],
            ['L1',      'B7',   967,    100,    'Z'],
            ['L2',      'B9',   1767,   100,    'Z'],
        ],

        'shunts': [
            ['name',    'bus',  'V_n',  'Q',    'model'],
            ['C1',      'B7',   230,    250,    'Z'],
            ['C2',      'B9',   230,    300,    'Z'],
        ],

        'generators': {
            'GEN': [
                ['name',    'bus',  'S_n',  'V_n',  'P',    'V',    'H',    'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['G1',      'B1',   900,    20,     700,    1.03,   6.5,    0,      1.8,    1.7,    0.3,        0.55,       0.25,       0.25,       8.0,        0.4,        0.03,       0.05],
                ['G2',      'B2',   900,    20,     700,    1.01,   6.5,    0,      1.8,    1.7,    0.3,        0.55,       0.25,       0.25,       8.0,        0.4,        0.03,       0.05],
                ['G3',      'B3',   900,    20,     719,    1.03,   6.175,  0,      1.8,    1.7,    0.3,        0.55,       0.25,       0.25,       8.0,        0.4,        0.03,       0.05],
                ['G4',      'B4',   900,    20,     700,    1.01,   6.175,  0,      1.8,    1.7,    0.3,        0.55,       0.25,       0.25,       8.0,        0.4,        0.03,       0.05],
            ]
        },

        'gov': {
            'TGOV1': [
                ['name', 'gen', 'R', 'D_t', 'V_min', 'V_max', 'T_1', 'T_2', 'T_3'],
                ['GOV3', 'G3', 0.12, 0.02, 0, 2, 0.1, 0.1, 0.3],
                ['GOV4', 'G4', 0.12, 0.02, 0, 2, 0.1, 0.1, 0.3],
            ],
            'HYGOV': [
                ['name', 'gen', 'R', 'r', 'T_f', 'T_r', 'T_g', 'A_t', 'T_w', 'q_nl', 'D_turb', 'G_min', 'V_elm',
                 'G_max', 'P_N'],
                ['HYGOV1', 'G1', 0.06, 0.5, 0.05, 5, 0.2, 1, 1, 0.01, 0.01, 0.0, 100, 2, 0],
                ['HYGOV2', 'G2', 0.06, 0.5, 0.05, 5, 0.2, 1, 1, 0.01, 0.01, 0.0, 100, 2, 0],
            ],
        },

        'avr': {
            'SEXS': [
                ['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max'],
                ['AVR1', 'G1', 100, 0.5, 3.0, 0.1, -3, 6],
                ['AVR2', 'G2', 100, 0.5, 3.0, 0.1, -3, 6],
                ['AVR3', 'G3', 100, 0.5, 3.0, 0.1, -3, 6],
                ['AVR4', 'G4', 100, 0.5, 3.0, 0.1, -3, 6],
            ]
        },

        'pss': {

        },
    }