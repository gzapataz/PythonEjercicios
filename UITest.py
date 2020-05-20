import ui
from scene import *

v = ui.load_view()
t = ui.Label = 'prueba'
v.border_width = 10
v.border_color = 'blue'
v.bg_color = 'pink'
v.multitouch_enabled
v.subviews
v.present('sheet')


class myclass(Scene):
	
	def draw(self):
		circle = ui.Path.oval(0,0,690,690)
		circle.line_width = 10
		self.face = ShapeNode(circle, 'yellow', 'black')
		self.add_child(self.face)
		
run(myclass())
