import os,sys,mystring

def flatten(column_header):
	if isinstance(column_header, list) or isinstance(column_header, tuple):
		return ' '.join([str(ch) for ch in column_header]).strip()
	return column_header


def generate_subplots_positions(num_plots, max_rows=None, max_cols=None):
	if max_rows is None and max_cols is None:
		max_rows = int(num_plots ** 0.5)
		max_cols = int(num_plots / max_rows) + (num_plots % max_rows > 0)
	elif max_rows is None:
		max_rows = int(num_plots / max_cols) + (num_plots % max_cols > 0)
	elif max_cols is None:
		max_cols = int(num_plots / max_rows) + (num_plots % max_rows > 0)

	positions = []
	for i in range(num_plots):
		row = i // max_cols + 1
		col = i % max_cols + 1
		print(f"[{row}, {col}]")
		positions += [(row,col)]

	return {
		"positions":positions,
		"rows":max_rows,
		"cols":max_cols,
	}

def update_fig(figure, **kwargs):
	success = False
	try:
		try:
			figure.update(**kwargs)
			success = True
		except Exception as e:
			import os,sys
			_, _, exc_tb = sys.exc_info();fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(":> Hit an unexpected error |figure.update| {0} @ {1}:{2}".format(e, fname, exc_tb.tb_lineno))

		try:
			figure.update_layout(**kwargs)
			success = True
		except Exception as e:
			import os,sys
			_, _, exc_tb = sys.exc_info();fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(":> Hit an unexpected error |figure.update_layout| {0} @ {1}:{2}".format(e, fname, exc_tb.tb_lineno))
	except:pass
	return success

common_defaults = {
	'Font': 'Times New Roman',
	'Font_Size':26,
	'Font_Color':'black',
	'DiscreteColours':['#f0f9e8', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac'],
	'DiscretePatterns':['x', '.', '+', '/', '-', '|', '^'],
}

from abc import ABC, abstractmethod
from copy import deepcopy as dc
class styleapplicator(ABC):
	def __init__(self):
		super().__init__()
		self.DiscreteColours = common_defaults['DiscreteColours']
		self.DiscretePatterns = common_defaults['DiscretePatterns']
	def assign_discrete_colormap(self, *keys):
		key_list = list(keys)
		if len(key_list) > len(self.DiscreteColours):
			raise Exception("There are too many keys, there are only {0} colours".format(len(self.DiscreteColours)))
		return {
			key:self.DiscreteColours[key_itr]
			for key_itr, key in enumerate(key_list)
		}
	def assign_discrete_patternmap(self, *keys):
		key_list = list(keys)
		if len(key_list) > len(self.DiscretePatterns):
			raise Exception("There are too many keys, there are only {0} colours".format(len(self.DiscretePatterns)))
		return self.DiscretePatterns[:len(key_list)-1]
	def assign_extras(self, *extras, keys=[]):
		output = {}
		extra_list = list(extras)
		if len(extra_list)  == 0:
			extra_list = ["patterns", "colours"]
		for extra_item in extra_list:
			if extra_item == "colours":
				output['color_discrete_map'] = self.assign_discrete_colormap(*keys)
			if extra_item == "patterns":
				output['pattern_shape_sequence'] = self.assign_discrete_patternmap(*keys)
		return output
	@staticmethod
	def clr():
		try:
			from IPython.display import clear_output;
			clear_output();
		except:pass
	def clear_screen(self):
		styleapplicator.clr();
	def reset(self):
		try:
			from IPython import get_ipython;
			ipython = get_ipython();
		except:pass
	@abstractmethod
	def __enter__(self):
		pass
	@abstractmethod
	def __exit__(self,*args, **kwargs):
		pass
	@abstractmethod
	def __call__(self, some_figure_obj):
		pass			

def plt_default(default_theme="seaborn"):
	import plotly.io as pio
	pio.templates.default = default_theme

def plt_style(**kwargs):
	font_keys = ['layout.font.family', 'layout.legend.font.family']
	font_size_keys = ['layout.font.size', 'layout.legend.font.size']

	style_dict = {
		'theme':'seaborn',
		'layout.plot_bgcolor': 'rgba(0, 0, 0, 0)',
		'layout.font.family': common_defaults['Font'],
		'layout.font.size': common_defaults['Font_Size'],
		'layout.xaxis.linecolor': 'black',
		'layout.xaxis.ticks': 'outside',
		'layout.xaxis.mirror': True,
		'layout.xaxis.showline': True,
		'layout.xaxis.type':'-',#https://plotly.com/python/reference/layout/xaxis/#layout-xaxis-type
		'layout.yaxis.linecolor': 'black',
		'layout.yaxis.ticks': 'outside',
		'layout.yaxis.mirror': True,
		'layout.yaxis.showline': True,
		'layout.autosize': True,
		'layout.showlegend': True,
		'layout.legend.bgcolor': 'rgba(0, 0, 0, 0)',
		'layout.legend.x': 1,
		'layout.legend.font.family': common_defaults['Font'],
		'layout.legend.font.size': common_defaults['Font_Size'],
		# Specialized:
		# 'layout.xaxis.range': (2.3, 2.5),
		'layout.yaxis.range': (0, +50),
		'layout.xaxis.title': r'$x$', #Latex Style
		'layout.yaxis.title': r'$y$', #Latex Style
		'layout.title': 'Advanced Example of a Line Plot with Plotly',
		'layout.title.xanchor': 'center',
		'layout.title.yanchor': 'top',
	}

	if "theme" in style_dict:
		plt_default(style_dict['theme'])
		del style_dict['theme']
	else:
		plt_default()

	if "all_font_size" in style_dict:
		del style_dict["all_font_size"]

	for key,value in kwargs.items():
		key=pltstyle.key_fix(key)
		if key == 'font':
			for font_key in font_keys:
				style_dict[font_key] = value
		elif key == 'size':
			for font_size_key in font_size_keys:
				style_dict[font_size_key] = value
		elif key == 'xlabel':
			style_dict['layout.xaxis.title'] = value
		elif key == 'ylabel':
			style_dict['layout.yaxis.title'] = value
		elif key == 'title':
			style_dict['layout.title'] = value
		elif key == 'title.font.family':
			style_dict['layout.title.font.family'] = value
		elif key == 'title.text':
			style_dict['layout.title.text'] = value
		elif key == 'overall.title':
			style_dict['layout.title.text'] = value
		elif key == 'title.font.size':
			style_dict['layout.title.font.size'] = value
		elif key.startswith('subplot.'):
			pass
		else:
			style_dict[key] = value
	return style_dict

class pltstyle(styleapplicator):
	def __init__(self, **kwargs):
		super().__init__()
		total_items = plt_style(**kwargs)
		self.all_font_size = None if "all_font_size" not in total_items else total_items['all_font_size']
		self.parent_plot = False if "parent_plot" not in total_items else total_items['parent_plot']
		reserved_words = ['majorplotkey', 'env_name', 'majorplotkey', 'parent_plot']
		self.kwargs = {self.self_key_fix(key):value for key,value in total_items.items() if not key.startswith('subplot') and key not in []}
		self.subplot_kwargs = {self.self_key_fix(key):value for key,value in total_items.items() if key.startswith('subplot') and key not in ['majorplotkey', 'env_name']}
		self.env_name = "base_style" if "env_name" not in total_items else total_items['env_name']
		self.majorplotkey = [] if "majorplotkey" not in total_items else total_items['majorplotkey']
		self.set_env()
	def __enter__(self):return self.from_env()
	def __exit__(self,*args, **kwargs):self.set_env();pass
	@staticmethod
	def key_fix(string, parent_plot=False):return string.replace('_','.') if not parent_plot else string.replace('.','_')
	def self_key_fix(self, string):
		return pltstyle.key_fix(string, parent_plot=self.parent_plot)
	def __call__(self, some_figure_obj):
		#https://plotly.com/python/subplots/
		key_filter_lambda = lambda x:True if self.majorplotkey == [] else lambda x:x in self.majorplotkey
		update_dicts = self.of(use_main_plot=True, key_filter=key_filter_lambda)
		#This seems to only work for plotly.go objects?
		update_fig(
			figure=some_figure_obj,
			**update_dicts
		)
		if self.all_font_size:
			self.set_all_font_sizes(figure=some_figure_obj, font_size=self.all_font_size)

		#Changing some per-plot settings since they're traces & annotations
		for key,value in self.subplot_kwargs.items():
			settings = {}
			if key == "subplot.title.font.size":
				settings['size'] = value
			elif key == "subplot.title.font.color":
				settings['color'] = value
			elif key == "subplot.title.font.family":
				settings['family'] = value

				#Setting all of the titles, they're all annotations?
				#https://github.com/plotly/plotly.py/issues/985
				#https://plotly.com/python/reference/layout/yaxis/
				try:
					for i in some_figure_obj['layout']['annotations']:
						i['font'] = settings
				except Exception as e:
					import os,sys
					_, _, exc_tb = sys.exc_info();fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
					print(":> Hit an unexpected error |some_figure_obj['layout']['annotations']| {0} @ {1}:{2}".format(e, fname, exc_tb.tb_lineno))

		self.clear_screen()
		return some_figure_obj
	def __getitem__(self, key):
		key=self.self_key_fix(key)
		if key in self.kwargs:
			return self.kwargs[key]
		elif key in self.subplot_kwargs:
			return self.subplot_kwargs[key]
		return None
	def __setitem__(self, key, value):
		key=self.self_key_fix(key)
		if not key.startswith('subplot'):
			self.kwargs[key]=value
		else: #key.startswith('subplot'):
			self.subplot_kwargs[key]=value
	def __delitem__(self, key):
		key=self.self_key_fix(key)
		if key in self.kwargs:
			del self.kwargs[key]
		elif key in self.subplot_kwargs:
			del self.subplot_kwargs[key]
	def __iter__(self):return iter(self.of(use_main_plot=None).values())
	def __reversed__(self):return reversed(self.of(use_main_plot=None).values())
	def __contains__(self, item):return item in self.of(use_main_plot=None).keys()
	def items(self, key_filter=lambda x:True):return [(x, y) for x,y in self.of(use_main_plot=None).items() if key_filter(x)]
	def keys(self, key_filter=lambda x:True):return [x for x in self.of(use_main_plot=None).keys() if key_filter(x)]
	def values(self, key_filter=lambda x:True):return [self[x] for x in self.of(use_main_plot=None).keys() if key_filter(x)]
	def of(self, use_main_plot=True, key_filter=lambda x:True):
		if use_main_plot is None:
			total_items = {
				**self.kwargs,
				**self.subplot_kwargs,
			}
			return {self.self_key_fix(x):y for x,y in total_items.items() if key_filter(x)}
		elif use_main_plot:
			return {self.self_key_fix(x):y for x,y in self.kwargs.items() if key_filter(x)}
		else:
			return {self.self_key_fix(x):y for x,y in self.subplot_kwargs.items() if key_filter(x)}
	@property
	def to_json(self):
		import json
		return json.dumps(self.of(use_main_plot=None))
	@staticmethod
	def from_json(json_string):
		import json
		return pltstyle(**json.loads(json_string))
	def set_env(self):
		import os,json
		os.environ[self.env_name] = self.to_json
	@staticmethod
	def from_env(*args, **kwargs):
		import os,json
		if 'env_name' not in kwargs:
			env_name = "base_style"
		else:
			env_name = kwargs['env_name']

		if env_name not in os.environ:
			pltstyle(**kwargs).set_env()

		current = pltstyle.from_json(os.environ[env_name])
		for key,value in kwargs.items():
			current[key] = value
		return current

	@staticmethod
	def set_all_font_sizes(figure, font_size=common_defaults['Font_Size']):
		update_fig(figure,
			title_font=dict(size=font_size),  # Title font size
			font=dict(size=font_size),  # General font size for all text
			xaxis=dict(title_font=dict(size=font_size), tickfont=dict(size=font_size)),  # X-axis title and tick font sizes
			yaxis=dict(title_font=dict(size=font_size), tickfont=dict(size=font_size)),  # Y-axis title and tick font sizes
			legend=dict(font=dict(size=font_size)),  # Legend font size
		)

	@staticmethod
	def wrap_args(figure, **kwargs):
		for key,value in kwargs.items():
			key=str(key)
			if update_fig(
				figure,
				**{key:value}
			):
				setattr(figure, "_"+key, value)