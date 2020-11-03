import os
import glob
import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import ast

paths = glob.glob("../CenterNet/reports/*")
experiments = [exp_path.split("/")[-1] for exp_path in paths]


def gather_results():
	header = ["AP","AP0.5","AP0.95","APM","APL","AR","AR0.5","AR0.95","ARM","ARL"]
	data = {}
	opts = {}
	for exp in experiments:
		reports_dir = "../CenterNet/reports/{}".format(exp)
		reports = glob.glob(reports_dir+"/*.csv")
		opt_path = "../CenterNet/reports/{}/train_opt.txt".format(exp)
		if os.path.isfile(opt_path):
			opts[exp] = get_opt_file(opt_path)
		for rep in reports:
			model_epoch = rep.split("/")[-1].split(".")[0].split("_")[-1]
			model_arch = rep.split("_")[0]
			model_domain = rep.split("_")[2]			
			with open(rep,"r") as csvFile:
				reader = csv.reader(csvFile)
				scores = map(float,next(reader))
				data[(exp,model_epoch)] = scores
	df = pd.DataFrame(data).T
	df.columns = header
	model = df.index.get_level_values(0)
	epochs = df.index.get_level_values(1).astype(float)
	df["epoch"] = epochs
	df.index = model
	df['arch'] = df.apply(lambda row: row.name.split("_")[0],axis=1)
	df['regime'] = df.apply(lambda row: row.name.split("_")[1],axis=1)
	df['domain'] = df.apply(lambda row: row.name.split("_")[2],axis=1)
	return df, opts


def plot_normal(scores, experiment_id):
	scores_experiment = scores.loc[experiment_id]
	arch = scores_experiment['arch'][0]
	regime = scores_experiment['regime'][0]
	domain = scores_experiment['domain'][0]
	fig,ax = plt.subplots(1,1)
	scores_experiment.plot(x='epoch',y='AP',style='*',ax=ax,
		title="Experiment: {}".format(experiment_id))
	ax.set_ylabel('AP')
	ax.set_xlabel("Epoch")
	ax.grid("minor")
	plt.savefig("../figures/{}_{}/AP_{}.png".format(arch,regime,experiment_id))

	fig,ax = plt.subplots(1,1)
	scores_experiment.plot(x='epoch',y='AR',style='*',ax=ax,
		title="Experiment: {}".format(experiment_id))
	ax.set_ylabel('AR')
	ax.set_xlabel("Epoch")
	ax.grid("minor")
	plt.savefig("../figures/{}_{}/AR_{}.png".format(arch,regime,experiment_id))


def plot_score_diff_domains(scores, score_name, model, regime):
	experiment = "{}_{}".format(model,regime)
	scores_dla_1x_color = scores.loc["{}_color".format(experiment)]
	scores_dla_1x_thermal = scores.loc[f"{experiment}_thermal"]
	fig, ax = plt.subplots(1,1)
	scores_dla_1x_color.plot(x='epoch',y=score_name,style='*',ax=ax,
		title="{}: {} {} Color vs Gray".format(score_name, model,regime),label='Color Images')
	scores_dla_1x_thermal.plot(x='epoch',y=score_name,style="*",ax=ax,label='Gray Images')
	ax.set_ylabel(score_name)
	ax.set_xlabel("Epoch")
	ax.grid("minor")
	plt.savefig("../figures/{}_{}/{}_color_vs_gray.png".format(model,regime,score_name))

def plot_scores_by_lr(scores,score_name,model,regime,domain):
	ids = ["bs32_mb8_lr5e-4","bs32_mb8_lr5e-3","bs32_mb8_lr5e-5","bs32_mb8_lr5e-5"]
	experiments =["{}_{}_{}_finetune_{}".format(model,regime,domain,lr) for lr in ids]
	fig, ax = plt.subplots(1,1)
	for exp in experiments:
		scores_lr = scores.loc[exp]
		lr = exp.split("_")[-1]
		scores_lr.plot(x='epoch',y=score_name,style='*',ax=ax,
			title="{}: {} {} Different LRs".format(score_name,model,regime),label="lr={}".format(lr))
	ax.set_ylabel(score_name)
	ax.set_xlabel("Epoch")
	ax.grid("minor")
	plt.savefig("../figures/{}_{}/{}_{}_finetune_lr".format(model,regime,score_name,domain))

def plot_epoch_test(scores,score_name,model,regime,domain,opts=None):
	#ids = ['50e_bs16_lr5e-4','50e_bs16_lr5e-4_wd35-45','50e_lr5e-4_wd40-48']
	#ids = ['50e_bs32_mb8_lr5e-4','50e_bs32_mb8_lr5e-4_wd35-45','50e_bs32_mb8_lr5e-4_wd40-47']
	# ids = ['50e_bs16_lr5e-4_wd40-48','50e_bs16_lr5e-4_wd35-45','50e_bs16_lr5e-4']
	ids = ['20e','20e_wd10-17','20e_wd15-18']
	#ids = ['20e','20e_wd10-17','20e_wd15-18','30e_wd15-18']
	fig, ax = plt.subplots(1,1)
	for exp_id in ids:
		experiment = "{}_{}_{}_finetune_{}".format(model,regime,domain,exp_id)
		opt = opts[experiment]
		scores_epoch = scores.loc[experiment]
		if opt['lr_step'][0] < int(opt['num_epochs']):
			scores_epoch.plot(x='epoch',y=score_name,style="*",ax=ax,label="wd = {}".format(opt['lr_step']))
		else:
			scores_epoch.plot(x='epoch',y=score_name,style="*",ax=ax,label="No wd")
	plt.title('{}: {} {} Different LR steps (init_lr={})'.format(score_name,model,regime,opt['lr']))
	ax.set_ylabel(score_name)
	ax.set_xlabel("Epoch")
	ax.grid("minor")
	plt.savefig("../figures/{}_{}/{}_{}_epoch_test".format(model,regime,score_name,domain))

def plot_batch_test(scores,score_name,model,regime,domain,opts=None):
	preamble = "backbonefrz"
	#ids = ['bs56_mb8_lr5e-5','bs32_mb8_lr5e-5','bs16_mb8_lr5e-5','bs8_mb8_lr5e-5']
	#ids = ['bs32_mb8_5e-4','bs32_mb8_lr5e-4','bs16_mb8_lr5e-4','bs8_mb8_lr5e-4']
	#ids = ['bs56_mb8_lr5e-3','bs32_mb8_lr5e-3','bs16_mb8_lr5e-3','bs8_mb8_lr5e-3']
	#ids = ['bs24_mb4_lr2.5e-3','bs16_mb4_lr2.5e-3','bs8_mb4_lr2.5e-3','bs4_mb4_lr2.5e-3']
	#ids = ['bs24_mb4_lr2.5e-4','bs16_mb4_lr2.5e-4','bs8_mb4_lr2.5e-4','bs4_mb4_lr2.5e-4']
	#ids = ['bs24_mb4_lr2.5e-5','bs16_mb4_lr2.5e-5','bs8_mb4_lr2.5e-5','bs4_mb4_lr2.5e-5']
	ids = ['mb8_bs56_lr5e-5','mb8_bs32_lr5e-5','mb8_bs16_lr5e-5','mb8_bs8_lr5e-5']
	fig, ax = plt.subplots(1,1)
	for exp_id in ids:
		experiment = "{}_{}_{}_{}_{}".format(model,regime,domain,preamble,exp_id)
		opt = opts[experiment]
		scores_epoch = scores.loc[experiment]
		lr = opt['lr']
		scores_epoch.plot(x='epoch',y=score_name,style="*",ax=ax,
			title='{}: {} {} (lr={}) Different Batch Sizes'.format(score_name, model,regime,lr),
			label="batch_size = {}".format(opt['batch_size']))
	lr = "{:.0e}".format(float(lr))
	ax.set_ylabel(score_name)
	ax.set_xlabel("Epoch")
	ax.grid("minor")
	plt.savefig("../figures/{}_{}/{}_{}_{}_{}_batch_test".format(model,regime,score_name,domain,lr,preamble))

def get_opt_file(opt_path):
	opts = []
	with open(opt_path,'r') as file:
		reader = csv.reader(file,delimiter="\n")
		for row in reader:
			row = row[0].lstrip()
			opts.append(row)
	opts = opts[5:]
	opts_dict = {item.split(":")[0]:item.split(":")[1].lstrip() for item in opts}
	opts_dict['lr_step'] = ast.literal_eval(opts_dict['lr_step'])
	return opts_dict


if __name__=='__main__':
	scores, opts = gather_results()

	#plot_batch_test(scores,'AP','dla','3x','gray',opts=opts)
	#plot_batch_test(scores,'AR','dla','3x','gray',opts=opts)


	scores = scores.set_index('epoch',append=True)
	AP_top_scores = scores.loc[scores['AP'].groupby(level=0).idxmax()][['AP']].reset_index(level=[1]).rename(columns={"epoch":"best_AP_epoch"})
	AR_top_scores = scores.loc[scores['AR'].groupby(level=0).idxmax()][['AR']].reset_index(level=[1]).rename(columns={"epoch":"best_AR_epoch"})
	top_scores = pd.merge(AP_top_scores,AR_top_scores,left_index=True,right_index=True)
	top_scores.to_csv('../results/top_scores.csv')

	top_ones = scores[['arch','domain','AP','AR']].groupby(['arch','domain']).max().loc[[('dla','color'),('dla','thermal'),('hg','thermal')]]
	top_ones = top_ones.reset_index()
	top_ones['experiment'] = top_ones.apply(lambda row: row['arch']+"_" + row['domain'],axis=1)
	fig,ax = plt.subplots(1,1)
	print(top_ones)
	"""
	plot_score(scores,'AR','dla','1x')
	plot_score(scores,'AP','dla','1x')
	
	plot_score(scores,'AP','hg','1x')
	plot_score(scores,'AR','hg','1x')
	"""

