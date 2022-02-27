I uploaded the postprocessing scripts that are used to generate the plots and table in our paper, https://arxiv.org/abs/2202.01954, to the postprocess_multitaskingpaper branch in my fork (https://github.com/pzhanggit/HydraGNN/tree/postprocess_multitaskingpaper/examples/postprocess). 
Inside examples/postprocess folder, Scatterplot_directcomparision_MLT_Comparison_*.py are to generate scatter plots.
ErrorPDF_MLT_SLT_Comparison.py is to generate error distribution pkl files for all individual runs.
ErrorPDF_MLT_SLT_Comparison_ensemble*.py is to read the pkl files generated in 2 and get the ensemble values and make the final error distribution plots.
Error_table_MLT_SLT_Comparison_ensemble.py file is to read the pkl files generated in 2 and output the RSME error table with standard deviations.
Losshistory_MLT_SLT_Comparison_*.py files are used to plot loss history in training. 
