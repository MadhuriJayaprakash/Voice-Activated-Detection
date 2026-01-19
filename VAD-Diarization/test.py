import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            # Original variables for SWH prediction
            'significant_height_of_combined_wind_waves_and_swell',  # SWH
            'total_precipitation',                                    # TP
            '10m_u_component_of_wind',                               # U10
            '10m_v_component_of_wind',                               # V10
            
            # Additional wave parameters
            'mean_wave_period',                                      # MWP - average time between waves
            'mean_zero_crossing_wave_period',                        # MP2 - alternative wave period measure
            'peak_wave_period',                                      # PP1D - period of dominant waves
            'mean_wave_direction',                                   # MWD - direction waves travel
        ],
        'year': [str(year) for year in range(1993, 2021)],
        'month': ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11', '12'],
        'time': '00:00',
        'area': [20, 65, 5, 80],  # North, West, South, East for India west coast
        'format': 'netcdf',
    },
    'complete_wave_data.nc')
