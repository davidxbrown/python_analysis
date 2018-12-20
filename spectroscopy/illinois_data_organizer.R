library(tidyverse)
library(rio)


spectrometer_data_ground_truth <- import_list('~/Downloads/PP2016-01447DR1_Supplemental_Dataset_1 (1).xlsx',
                                             skip = 2,
                                             setclass = 'tibble')


chlorophyll <- spectrometer_data_ground_truth$Chlorophyll
nitrogen <- spectrometer_data_ground_truth$`Percent N`
SLA <- spectrometer_data_ground_truth$'SLA'
Sucrose <- spectrometer_data_ground_truth$Sucrose
Vmax <-  spectrometer_data_ground_truth$Vmax



full_table <- full_join(full_join(full_join(full_join(chlorophyll, nitrogen), SLA), Vmax), Sucrose)
full_table_view <- select(full_table, measured_percentN, Measured_Chl, measured_SLA, measured_Vmax, Measured_Sucrose, Wave_500, Wave_1600, Wave_2400)

chl_n_table <- inner_join(chlorophyll, nitrogen)
chl_n_table_view <- select(chl_n_table, measured_percentN, Measured_Chl, Wave_500, Wave_1600, Wave_2400)


write.csv(full_table, file = "/Users/doctorfarmerbrown/git/spectroscopy/Illinois_spectral_data.csv")
write.csv(full_table_view, file = "/Users/doctorfarmerbrown/git/spectroscopy/Illinois_spectral_data_viewable.csv")
write.csv(chl_n_table, file = "/Users/doctorfarmerbrown/git/spectroscopy/Illinois_chl_n_spectral_data.csv")
write.csv(chl_n_table_view, file = "/Users/doctorfarmerbrown/git/spectroscopy/Illinois_chl_n_spectral_data_viewable.csv")

save(full_table, 
     full_table_view, 
     chl_n_table,
     chl_n_table_view,
     file = '/Users/doctorfarmerbrown/git/spectroscopy/illinois_spectroscopy.RData')
