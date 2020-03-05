"""The definitions of the things we need to aggregate each plant part."""

import pudl

plant_parts = {
    'plant': {
        'id_cols': ['plant_id_eia'],
        'denorm_table': None,
        'denorm_cols': None,
        'install_table': None,
        'false_grans': None,
        'ag_cols': {
            'total_fuel_cost': 'sum',
            'net_generation_mwh': 'sum',
            'capacity_mw': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
        },
    },
    'plant_gen': {
        'id_cols': ['plant_id_eia', 'generator_id'],
        # unit_id_pudl are associated with plant_ids & plant_ids/generator_ids
        'denorm_table': None,
        'denorm_cols': None,
        'install_table': None,
        'false_grans': ['plant', 'plant_unit'],
        'ag_cols': {
            'capacity_mw': pudl.helpers.sum_na,
            'net_generation_mwh': 'sum',
            'total_fuel_cost': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
        },
        'ag_tables': {
            'generation_eia923': {
                'denorm_table': None,
                'denorm_cols': None,
                'ag_cols': {
                    'net_generation_mwh': 'sum',
                },
                'wtavg_cols': None,
            },
            'generators_eia860': {
                'denorm_table': None,
                'denorm_cols': None,
                'ag_cols': {
                    'capacity_mw': 'sum',
                },
                'wtavg_cols': None,
            },
            'mcoe': {
                'denorm_table': None,
                'denorm_cols': None,
                'ag_cols': {
                    'total_fuel_cost': 'sum',
                    'total_mmbtu': 'sum'
                },
                'wtavg_cols': {
                    'fuel_cost_per_mwh': 'capacity_mw',  # 'wtavg_mwh',
                    'heat_rate_mmbtu_mwh': 'capacity_mw',  # 'wtavg_mwh',
                    'fuel_cost_per_mmbtu': 'capacity_mw',  # 'wtavg_mwh',
                },

            }
        },
    },
    'plant_unit': {
        'id_cols': ['plant_id_eia', 'unit_id_pudl'],
        # unit_id_pudl are associated with plant_ids & plant_ids/generator_ids
        'denorm_table': 'boiler_generator_assn_eia860',
        'denorm_cols': ['plant_id_eia', 'generator_id', 'report_date'],
        'install_table': 'boiler_generator_assn_eia860',
        'false_grans': ['plant'],
        'ag_cols': {
            'capacity_mw': 'sum',
            'net_generation_mwh': 'sum',
            'total_fuel_cost': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
        },
    },
    'plant_technology': {
        'id_cols': ['plant_id_eia', 'technology_description'],
        'denorm_table': 'generators_eia860',
        'denorm_cols': ['plant_id_eia', 'generator_id', 'report_date'],
        'install_table': 'generators_eia860',
        'false_grans': ['plant_prime_mover', 'plant_gen', 'plant_unit', 'plant'
                        ],
        'ag_cols': {
            'capacity_mw': 'sum',
            'net_generation_mwh': 'sum',
            'total_fuel_cost': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
        },
    },
    'plant_prime_fuel': {
        'id_cols': ['plant_id_eia', 'energy_source_code_1'],
        'denorm_table': 'generators_eia860',
        'denorm_cols': ['plant_id_eia', 'generator_id', 'report_date'],
        'install_table': 'generators_eia860',
        'false_grans': ['plant_technology', 'plant_prime_mover', 'plant_gen',
                        'plant_unit', 'plant'],
        'ag_cols': {
            'capacity_mw': 'sum',
            'net_generation_mwh': 'sum',
            'total_fuel_cost': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
        },
    },
    'plant_prime_mover': {
        'id_cols': ['plant_id_eia', 'prime_mover_code'],
        'denorm_table': 'generators_entity_eia',
        'denorm_cols': ['plant_id_eia', 'generator_id'],
        'install_table': None,
        'false_grans': ['plant_gen', 'plant_unit', 'plant'],
        'ag_cols': {
            'capacity_mw': 'sum',
            'net_generation_mwh': 'sum',
            'total_fuel_cost': 'sum',
            'total_mmbtu': 'sum'
        },
        'wtavg_cols': {
            'fuel_cost_per_mwh': 'capacity_mw',
            'heat_rate_mmbtu_mwh': 'capacity_mw',
            'fuel_cost_per_mmbtu': 'capacity_mw',
        },
    }
}
