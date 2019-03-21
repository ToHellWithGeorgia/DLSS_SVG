#include "software_renderer.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include "assert.h"

#include "triangulation.h"

using namespace std;

namespace CS248 {


// Implements SoftwareRenderer //

// fill a sample location with color
void SoftwareRendererImp::fill_sample(int sx, int sy, const Color &color) {
  if (sx < 0 || sx >= w) return;
  if (sy < 0 || sy >= h) return;

  Color pixel_color;
  float inv255 = 1.0 / 255.0;
  pixel_color.r = sample_buffer[4 * (sx + sy * w)] * inv255;
  pixel_color.g = sample_buffer[4 * (sx + sy * w) + 1] * inv255;
  pixel_color.b = sample_buffer[4 * (sx + sy * w) + 2] * inv255;
  pixel_color.a = sample_buffer[4 * (sx + sy * w) + 3] * inv255;

  // pixel_color = ref->alpha_blending_helper(pixel_color, color);
  this->alpha_blending(pixel_color, color);

  sample_buffer[4 * (sx + sy * w)] = (uint8_t)(pixel_color.r * 255);
  sample_buffer[4 * (sx + sy * w) + 1] = (uint8_t)(pixel_color.g * 255);
  sample_buffer[4 * (sx + sy * w) + 2] = (uint8_t)(pixel_color.b * 255);
  sample_buffer[4 * (sx + sy * w) + 3] = (uint8_t)(pixel_color.a * 255);
}

void SoftwareRendererImp::alpha_blending(Color &pixel_color, const Color &color) {
  pixel_color.a = 1 - (1 - pixel_color.a) * (1 - color.a);
  pixel_color.r = (1 - color.a) * pixel_color.a * pixel_color.r + color.r * color.a;
  pixel_color.g = (1 - color.a) * pixel_color.a * pixel_color.g + color.g * color.a;
  pixel_color.b = (1 - color.a) * pixel_color.a * pixel_color.b + color.b * color.a; 
}

// fill samples in the entire pixel specified by pixel coordinates
void SoftwareRendererImp::fill_pixel(int x, int y, const Color &color) {

	// Task 2: Re-implement this function
  for (int i = sample_rate * x; i < sample_rate * x + sample_rate; ++i) {
    for (int j = sample_rate * y; j < sample_rate * y + sample_rate; ++j) {
      fill_sample(i, j, color);
    }
  }  

	// // check bounds
	// if (x < 0 || x >= target_w) return;
	// if (y < 0 || y >= target_h) return;

	// Color pixel_color;
	// float inv255 = 1.0 / 255.0;
	// pixel_color.r = render_target[4 * (x + y * target_w)] * inv255;
	// pixel_color.g = render_target[4 * (x + y * target_w) + 1] * inv255;
	// pixel_color.b = render_target[4 * (x + y * target_w) + 2] * inv255;
	// pixel_color.a = render_target[4 * (x + y * target_w) + 3] * inv255;

	// pixel_color = ref->alpha_blending_helper(pixel_color, color);

	// render_target[4 * (x + y * target_w)] = (uint8_t)(pixel_color.r * 255);
	// render_target[4 * (x + y * target_w) + 1] = (uint8_t)(pixel_color.g * 255);
	// render_target[4 * (x + y * target_w) + 2] = (uint8_t)(pixel_color.b * 255);
	// render_target[4 * (x + y * target_w) + 3] = (uint8_t)(pixel_color.a * 255);

}

void SoftwareRendererImp::draw_svg( SVG& svg ) {

  // set top level transformation
  transformation = canvas_to_screen;

  // Reset the sample buffer
  std::fill(this->sample_buffer.begin(), this->sample_buffer.end(), 255);

  // draw all elements
  for ( size_t i = 0; i < svg.elements.size(); ++i ) {
    draw_element(svg.elements[i]);
  }

  // draw canvas outline
  Vector2D a = transform(Vector2D(    0    ,     0    )); a.x--; a.y--;
  Vector2D b = transform(Vector2D(svg.width,     0    )); b.x++; b.y--;
  Vector2D c = transform(Vector2D(    0    ,svg.height)); c.x--; c.y++;
  Vector2D d = transform(Vector2D(svg.width,svg.height)); d.x++; d.y++;

  rasterize_line(a.x, a.y, b.x, b.y, Color::Black);
  rasterize_line(a.x, a.y, c.x, c.y, Color::Black);
  rasterize_line(d.x, d.y, b.x, b.y, Color::Black);
  rasterize_line(d.x, d.y, c.x, c.y, Color::Black);

  // resolve and send to render target
  resolve();

}

void SoftwareRendererImp::set_sample_rate( size_t sample_rate ) {

  // Task 2: 
  // You may want to modify this for supersampling support
  this->sample_rate = sample_rate;

  // Resize the super sample buffer and clear the sample_buffer
  //
  this->w = this->target_w * sample_rate;
  this->h = this->target_h * sample_rate;
  this->sample_buffer.resize(4 * this->w * this->h);
  
}

void SoftwareRendererImp::set_render_target( unsigned char* render_target,
                                             size_t width, size_t height ) {

  // Task 2: 
  // You may want to modify this for supersampling support
  this->render_target = render_target;
  this->target_w = width;
  this->target_h = height;

  // Resize the super sample buffer and clear the sample_buffer
  //
  this->w = this->target_w * sample_rate;
  this->h = this->target_h * sample_rate;
  this->sample_buffer.resize(4 * this->w * this->h);
}

void SoftwareRendererImp::draw_element( SVGElement* element ) {

	// Task 3 (part 1):
	// Modify this to implement the transformation stack
  Matrix3x3 temp = transformation;
  transformation = transformation * (element->transform);
	switch (element->type) {
	case POINT:
		draw_point(static_cast<Point&>(*element));
		break;
	case LINE:
		draw_line(static_cast<Line&>(*element));
		break;
	case POLYLINE:
		draw_polyline(static_cast<Polyline&>(*element));
		break;
	case RECT:
		draw_rect(static_cast<Rect&>(*element));
		break;
	case POLYGON:
		draw_polygon(static_cast<Polygon&>(*element));
		break;
	case ELLIPSE:
		draw_ellipse(static_cast<Ellipse&>(*element));
		break;
	case IMAGE:
		draw_image(static_cast<Image&>(*element));
		break;
	case GROUP:
		draw_group(static_cast<Group&>(*element));
		break;
	default:
		break;
	}
  transformation = temp;

}


// Primitive Drawing //

void SoftwareRendererImp::draw_point( Point& point ) {

  Vector2D p = transform(point.position);
  rasterize_point( p.x, p.y, point.style.fillColor );

}

void SoftwareRendererImp::draw_line( Line& line ) { 

  Vector2D p0 = transform(line.from);
  Vector2D p1 = transform(line.to);
  rasterize_line( p0.x, p0.y, p1.x, p1.y, line.style.strokeColor );

}

void SoftwareRendererImp::draw_polyline( Polyline& polyline ) {

  Color c = polyline.style.strokeColor;

  if( c.a != 0 ) {
    int nPoints = polyline.points.size();
    for( int i = 0; i < nPoints - 1; i++ ) {
      Vector2D p0 = transform(polyline.points[(i+0) % nPoints]);
      Vector2D p1 = transform(polyline.points[(i+1) % nPoints]);
      rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    }
  }
}

void SoftwareRendererImp::draw_rect( Rect& rect ) {

  Color c;
  
  // draw as two triangles
  float x = rect.position.x;
  float y = rect.position.y;
  float w = rect.dimension.x;
  float h = rect.dimension.y;

  Vector2D p0 = transform(Vector2D(   x   ,   y   ));
  Vector2D p1 = transform(Vector2D( x + w ,   y   ));
  Vector2D p2 = transform(Vector2D(   x   , y + h ));
  Vector2D p3 = transform(Vector2D( x + w , y + h ));
  
  // draw fill
  c = rect.style.fillColor;
  if (c.a != 0 ) {
    rasterize_triangle( p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, c );
    rasterize_triangle( p2.x, p2.y, p1.x, p1.y, p3.x, p3.y, c );
  }

  // draw outline
  c = rect.style.strokeColor;
  if( c.a != 0 ) {
    rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    rasterize_line( p1.x, p1.y, p3.x, p3.y, c );
    rasterize_line( p3.x, p3.y, p2.x, p2.y, c );
    rasterize_line( p2.x, p2.y, p0.x, p0.y, c );
  }

}

void SoftwareRendererImp::draw_polygon( Polygon& polygon ) {

  Color c;

  // draw fill
  c = polygon.style.fillColor;
  if( c.a != 0 ) {

    // triangulate
    vector<Vector2D> triangles;
    triangulate( polygon, triangles );

    // draw as triangles
    for (size_t i = 0; i < triangles.size(); i += 3) {
      Vector2D p0 = transform(triangles[i + 0]);
      Vector2D p1 = transform(triangles[i + 1]);
      Vector2D p2 = transform(triangles[i + 2]);
      rasterize_triangle( p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, c );
    }
  }

  // draw outline
  c = polygon.style.strokeColor;
  if( c.a != 0 ) {
    int nPoints = polygon.points.size();
    for( int i = 0; i < nPoints; i++ ) {
      Vector2D p0 = transform(polygon.points[(i+0) % nPoints]);
      Vector2D p1 = transform(polygon.points[(i+1) % nPoints]);
      rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    }
  }
}

void SoftwareRendererImp::draw_ellipse( Ellipse& ellipse ) {

  // Extra credit 
  float rx = ellipse.radius.x, ry = ellipse.radius.y;
  float cx = ellipse.center.x, cy = ellipse.center.y;

  // Get the edge point of x and y axis after transformation
  //
  Vector2D ct = transform(Vector2D(cx, cy));
  Vector2D xt = transform(Vector2D(cx - rx, cy));
  Vector2D yt = transform(Vector2D(cx, cy - ry));


  rasterize_ellipse(ct.x, ct.y, xt.x, xt.y, yt.x, yt.y,
                    ellipse.style.strokeColor,
                    ellipse.style.fillColor);

}

void SoftwareRendererImp::draw_image( Image& image ) {

  Vector2D p0 = transform(image.position);
  Vector2D p1 = transform(image.position + image.dimension);

  rasterize_image( p0.x, p0.y, p1.x, p1.y, image.tex );
}

void SoftwareRendererImp::draw_group( Group& group ) {

  for ( size_t i = 0; i < group.elements.size(); ++i ) {
    draw_element(group.elements[i]);
  }

}

// Rasterization //

// The input arguments in the rasterization functions 
// below are all defined in screen space coordinates

void SoftwareRendererImp::rasterize_point( float x, float y, Color color ) {

  // fill in the nearest pixel
  int sx = (int)floor(x);
  int sy = (int)floor(y);

  // check bounds
  if (sx < 0 || sx >= target_w) return;
  if (sy < 0 || sy >= target_h) return;

  // fill sample - NOT doing alpha blending!
  // TODO: Call fill_pixel here to run alpha blending
  // render_target[4 * (sx + sy * target_w)] = (uint8_t)(color.r * 255);
  // render_target[4 * (sx + sy * target_w) + 1] = (uint8_t)(color.g * 255);
  // render_target[4 * (sx + sy * target_w) + 2] = (uint8_t)(color.b * 255);
  // render_target[4 * (sx + sy * target_w) + 3] = (uint8_t)(color.a * 255);
  fill_pixel(sx, sy, color);
}

void SoftwareRendererImp::rasterize_line( float x0, float y0,
                                          float x1, float y1,
                                          Color color) {

  // Extra credit (delete the line below and implement your own)
  ref->rasterize_line_helper(x0, y0, x1, y1, target_w, target_h, color, this);

}

void SoftwareRendererImp::rasterize_triangle( float x0, float y0,
                                              float x1, float y1,
                                              float x2, float y2,
                                              Color color ) {
  // Task 1: 
  // Implement triangle rasterization (you may want to call fill_sample here)

  // Lambda function to test the line equation
  //
  auto LineEquationTest = [](float p0x, float p0y, float p1x, float p1y,
                             float x, float y) {
    float result = -(x - p0x) * (p1y - p0y) + (y - p0y) * (p1x - p0x);
    // TODO: Check if this should be < or <=
    //
    return result <= 0.0;
  };


  // Make sure P0-P1-P2 are in counter-clock-wise direction.
  //
  bool ccw = LineEquationTest(x0, y0, x1, y1, x2, y2);
  if (!ccw) {
    float temp_x = x1; 
    x1 = x2;
    x2 = temp_x;
    float temp_y = y1;
    y1 = y2;
    y2 = temp_y;
  }

  // Lambda function to test if a point is in a triangle
  //
  auto PointInTriangle = [=](float x, float y) {
    bool test0 = LineEquationTest(x0, y0, x1, y1, x, y);
    bool test1 = LineEquationTest(x1, y1, x2, y2, x, y);
    bool test2 = LineEquationTest(x2, y2, x0, y0, x, y);
    return test0 && test1 && test2;
  };

  // We start from the point with both of the other 2 points on its same side
  // (both direction), and use zigzag to traverse the shape.
  // xstart and ystart belong to the same point
  // xend and yend should be the greater of the rest 2
  //
  float xstart, ystart;
  float xend, yend;
  int xdir = 0, ydir = 0;

  // Find the start and end point of the triangle traversal
  //
  vector<float> xs{x0, x1, x2};
  vector<float> ys{y0, y1, y2};
  vector<float> xsorted(xs);
  vector<float> ysorted(ys);
  sort(xsorted.begin(), xsorted.end());
  sort(ysorted.begin(), ysorted.end());

  for (int i = 0; i < 3; ++i) {
    if (((xs[i] == xsorted[0]) || (xs[i] == xsorted[2])) &&
        ((ys[i] == ysorted[0]) || (ys[i] == ysorted[2]))) {
      xstart = xs[i];
      ystart = ys[i];
      break;
    }
  }

  if (xstart == xsorted[0]) {
    xend = xsorted[2];
    xdir = 1;
  } else if (xstart == xsorted[2]) {
    xend = xsorted[0];
    xdir = -1;
  }

  if (ystart == ysorted[0]) {
    yend = ysorted[2];
    ydir = 1;
  } else if (ystart == ysorted[2]) {
    yend = ysorted[0];
    ydir = -1;
  }

  assert(xdir != 0);
  assert(ydir != 0);

  // Start traversing the triangle.
  //
  int txs = xdir == 1 ? floor(xstart) - 1 : ceil(xstart) + 1;
  int txe = xdir == 1 ? ceil(xend) + 1 : floor(xend) - 1;
  int tys = ydir == 1 ? floor(ystart) - 1 : ceil(ystart) + 1;
  int tye = ydir == 1 ? ceil(yend) + 1 : floor(yend) - 1;
  
  for (int sx = txs; sx != txe; sx += xdir) {
    for (int sy = tys; sy != tye; sy += ydir) {
      // Boundary check is included in fill_sample
      // TODO: implement the zigzag
      //

      // Perform the supersampling
      //
      for (int si = 0; si < sample_rate; ++si) {
        for (int sj = 0; sj < sample_rate; ++sj) {
          float inc = 0.5 / sample_rate;
          if (PointInTriangle(sx + inc * (2 * si + 1), sy + inc * (2 * sj + 1))) {
            fill_sample(sx * sample_rate + si, sy * sample_rate + sj, color);
          }
        }
      }
    }
  }
}

void SoftwareRendererImp::rasterize_ellipse(float cx, float cy, 
                                            float xx, float xy,
                                            float yx, float yy,
                                            Color stroke_color,
                                            Color fill_color) {
  // Lambda to perform square
  //
  auto square = [] (float x) {
    return x * x;
  };

  // Calculate the transformed rx and ry
  //
  float rx = sqrt((xx - cx) * (xx - cx) + (xy - cy) * (xy - cy));
  float ry = sqrt((yx - cx) * (yx - cx) + (yy - cy) * (yy - cy));

  // Caculate the sin(theta) and cos(theta)
  //
  Vector2D dir_a = Vector2D(-1, 0);
  Vector2D dir_b = Vector2D(xx - cx, xy - cy);
  Vector2D dir_n = Vector2D(0, -1);
  float cosine = dot(dir_a, dir_b) / (dir_a.norm() * dir_b.norm());
  float sine = 0.0;

  if (dot(dir_n, dir_b) >= 0) {
    sine = sqrt(1 - square(cosine));
  } else {
    sine = - sqrt(1 - square(cosine));
  }

  // Test if the point is in the ellipse.
  //
  auto PointInEllipse = [=] (float x, float y) {
    return square((x - cx) * cosine + (y - cy) * sine) / square(rx) +  
           square((x - cx) * sine + (y - cy) * cosine) / square(ry) <= 1;
  };

  // We sample 4 vectices + 1 middle point of the pixel.
  // If not all the points are inside/outside of the ellipse
  // it's on the edge.
  //
  auto PointOnEllipse = [=] (float x, float y, float inc) {
    bool test1 = PointInEllipse(x - inc, y - inc);
    bool test2 = PointInEllipse(x - inc, y + inc);
    bool test3 = PointInEllipse(x + inc, y - inc);
    bool test4 = PointInEllipse(x + inc, y + inc);
    bool test5 = PointInEllipse(x, y);
    return !((test1 && test2 && test3 && test4 && test5) ||
             (!test1 && !test2 && !test3 && !test4 && !test5));
  };

  // In non axis-aligned ellipse, make sure we cover a bigger region
  //
  float dec = rx > ry ? rx + 5 : ry + 5;
  float xstart = cx - dec, ystart = cy - dec;
  float xend = cx + dec, yend = cy + dec;

  // Rasterize the fill
  for (int sx = floor(xstart); sx != ceil(xend); ++sx) {
    for (int sy = floor(ystart); sy != ceil(yend); ++sy) {
      // Perform the supersampling
      //
      for (int si = 0; si < sample_rate; ++si) {
        for (int sj = 0; sj < sample_rate; ++sj) {
          float inc = 0.5 / sample_rate;
          if (PointInEllipse(sx + inc * (2 * si + 1), sy + inc * (2 * sj + 1))) {
            fill_sample(sx * sample_rate + si, sy * sample_rate + sj, fill_color);
          }
        }
      }
    }
  }

  // Rasterize the outline
  for (int sx = floor(xstart); sx != ceil(xend); ++sx) {
    for (int sy = floor(ystart); sy != ceil(yend); ++sy) {
      // Perform the supersampling
      //
      for (int si = 0; si < sample_rate; ++si) {
        for (int sj = 0; sj < sample_rate; ++sj) {
          float inc = 0.5 / sample_rate;
          if (PointOnEllipse(sx + inc * (2 * si + 1), sy + inc * (2 * sj + 1), inc)) {
            fill_sample(sx * sample_rate + si, sy * sample_rate + sj, stroke_color);
          }
        }
      }
    }
  }
}

void SoftwareRendererImp::rasterize_image( float x0, float y0,
                                           float x1, float y1,
                                           Texture& tex ) {
  // Task 4: 
  // Implement image rasterization (you may want to call fill_sample here)
  int s_x0 = floor(x0), s_y0 = floor(y0);
  int s_x1 = ceil(x1), s_y1 = ceil(y1);

  for (int sx = s_x0; sx != s_x1; ++sx) {
    for (int sy = s_y0; sy != s_y1; ++sy) {
      // Perform the supersampling
      //
      for (int si = 0; si < sample_rate; ++si) {
        for (int sj = 0; sj < sample_rate; ++sj) {
          float inc = 0.5 / sample_rate;
          float u = (sx + inc * (2 * si + 1) - x0) / (x1 - x0);
          float v = (sy + inc * (2 * sj + 1) - y0) / (y1 - y0);
          if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
            Color sample_color = sampler->sample_bilinear(tex, u, v, 0);
            fill_sample(sx * sample_rate + si, sy * sample_rate + sj, sample_color);
          }
        }
      }
    }
  }
}

// resolve samples to render target
void SoftwareRendererImp::resolve( void ) {

  // Task 2: 
  // Implement supersampling
  // You may also need to modify other functions marked with "Task 2".

  for (int i = 0; i < target_w; i++) {
    for (int j = 0; j < target_h; j++) {
      // Use a simple box filter first
      //
      int r = 0, g = 0, b = 0, a = 0;
      for (int si = 0; si < sample_rate; ++si) {
        for (int sj = 0; sj < sample_rate; ++sj) {
          int sx = sample_rate * i + si;
          int sy = sample_rate * j + sj;
          r += sample_buffer[4 * (sx + sy * w)];
          g += sample_buffer[4 * (sx + sy * w) + 1];
          b += sample_buffer[4 * (sx + sy * w) + 2];
          a += sample_buffer[4 * (sx + sy * w) + 3];
        }
      }

      render_target[4 * (i + j * target_w)] = r / (sample_rate * sample_rate);
      render_target[4 * (i + j * target_w) + 1] = g / (sample_rate * sample_rate);
      render_target[4 * (i + j * target_w) + 2] = b / (sample_rate * sample_rate);
      render_target[4 * (i + j * target_w) + 3] = a / (sample_rate * sample_rate);

      // render_target[4 * (i + j * target_w)] = sample_buffer[4 * (i + j * target_w)];
      // render_target[4 * (i + j * target_w) + 1] = sample_buffer[4 * (i + j * target_w) + 1];
      // render_target[4 * (i + j * target_w) + 2] = sample_buffer[4 * (i + j * target_w) + 2];
      // render_target[4 * (i + j * target_w) + 3] = sample_buffer[4 * (i + j * target_w) + 3];
    }
  }
  // cout << "target width: " << target_w << " height: " << target_h << endl;
  return;

}


} // namespace CS248
