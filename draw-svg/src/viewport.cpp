#include "viewport.h"

#include "CS248.h"

namespace CS248 {

void ViewportImp::set_viewbox( float x, float y, float span ) {

  // Task 3 (part 2): 
  // Set svg to normalized device coordinate transformation. Your input
  // arguments are defined as SVG canvans coordinates.
  Matrix3x3 T = Matrix3x3::identity();
  Matrix3x3 S = Matrix3x3::identity();

  T(0, 2) = span - x;
  T(1, 2) = span - y;

  S(0, 0) = 1.0 / (2 * span);
  S(1, 1) = 1.0 / (2 * span);


  this->x = x;
  this->y = y;
  this->span = span; 

  set_canvas_to_norm(S * T);

}

void ViewportImp::update_viewbox( float dx, float dy, float scale ) { 
  
  this->x -= dx;
  this->y -= dy;
  this->span *= scale;
  set_viewbox( x, y, span );
}

} // namespace CS248
