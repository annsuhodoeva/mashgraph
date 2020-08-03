#include <iostream>
#include <cstdint>

#include <limits> 
#include <string>
#include <vector>
#include <unordered_map>
#define _USE_MATH_DEFINES 
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Bitmap.h"



void convert_xyz_to_cube_uv(float x, float y, float z, int *index, float *u, float *v)
{
  float absX = fabs(x);
  float absY = fabs(y);
  float absZ = fabs(z);
  
  int isXPositive = x > 0 ? 1 : 0;
  int isYPositive = y > 0 ? 1 : 0;
  int isZPositive = z > 0 ? 1 : 0;
  
  float maxAxis, uc, vc;
  
  // POSITIVE X
  if (isXPositive && absX >= absY && absX >= absZ) {
    // u (0 to 1) goes from +z to -z
    // v (0 to 1) goes from -y to +y
    maxAxis = absX;
    uc = -z;
    vc = y;
    *index = 0;
  }
  // NEGATIVE X
  if (!isXPositive && absX >= absY && absX >= absZ) {
    // u (0 to 1) goes from -z to +z
    // v (0 to 1) goes from -y to +y
    maxAxis = absX;
    uc = z;
    vc = y;
    *index = 1;
  }
  // POSITIVE Y
  if (isYPositive && absY >= absX && absY >= absZ) {
    // u (0 to 1) goes from -x to +x
    // v (0 to 1) goes from +z to -z
    maxAxis = absY;
    uc = x;
    vc = -z;
    *index = 2;
  }
  // NEGATIVE Y
  if (!isYPositive && absY >= absX && absY >= absZ) {
    // u (0 to 1) goes from -x to +x
    // v (0 to 1) goes from -z to +z
    maxAxis = absY;
    uc = x;
    vc = z;
    *index = 3;
  }
  // POSITIVE Z
  if (isZPositive && absZ >= absX && absZ >= absY) {
    // u (0 to 1) goes from -x to +x
    // v (0 to 1) goes from -y to +y
    maxAxis = absZ;
    uc = x;
    vc = y;
    *index = 4;
  }
  // NEGATIVE Z
  if (!isZPositive && absZ >= absX && absZ >= absY) {
    // u (0 to 1) goes from +x to -x
    // v (0 to 1) goes from -y to +y
    maxAxis = absZ;
    uc = -x;
    vc = y;
    *index = 5;
  }

  // Convert range from -1 to 1 to 0 to 1
  *u = 0.5f * (uc / maxAxis + 1.0f);
  *v = 0.5f * (vc / maxAxis + 1.0f);
}


class Vec2f {
	float a, b;
	public:
	Vec2f(float aa = 0, float bb = 0) : a(aa), b(bb) {}
	const Vec2f& operator=(const Vec2f& v) {
		a = v.a;
		b = v.b;
		return *this;
	}
	float get_a() const {
		return a;
	}
	float get_b() const {
		return b;
	}
	Vec2f operator+(Vec2f v) const {
		return Vec2f(a + v.a, b + v.b);
	}
	Vec2f operator-(Vec2f v) const {
		return Vec2f(a - v.a, b - v.b);
	}
	float operator*(Vec2f v) const {
		return a * v.a + b * v.b ;
	}
	float &operator[](int i) //перегрузка []     
    {
        if (i == 0) {
			return a;
		}
		else if (i == 1) {
			return b;
		}
    }
	Vec2f operator*(float d) const {
		return Vec2f(a * d, b * d);
	}
	Vec2f operator/(float d) const {
		return Vec2f(a / d, b / d);
	}
	float norm() {
		return sqrtf(a * a + b * b);
	}
	Vec2f normalize() {
		return *this / norm();
	}
};

class Vec3f {
	float a, b, c;
	public:
	Vec3f(float aa = 0, float bb = 0, float cc = 0) : a(aa), b(bb), c(cc) {}
	const Vec3f& operator=(const Vec3f& v) {
		a = v.a;
		b = v.b;
		c = v.c;
		return *this;
	}
	float get_a() const {
		return a;
	}
	float get_b() const {
		return b;
	}
	float get_c() const {
		return c;
	}
	float &operator[](int i) //перегрузка []     
    {
		if (i == 0) {
			return a;
		}
		else if (i == 1) {
			return b;
		}
		else if (i == 2) {
			return c;
		}
	}
	Vec3f operator+(Vec3f v) const {
		return Vec3f(a + v.a, b + v.b, c + v.c);
	}
	Vec3f operator-(Vec3f v) const {
		return Vec3f(a - v.a, b - v.b, c - v.c);
	}
	float operator*(Vec3f v) const {
		return a * v.a + b * v.b + c * v.c;
	}
	Vec3f operator*(float d) const {
		return Vec3f(a * d, b * d, c * d);
	}
	Vec3f operator/(float d) const {
		return Vec3f(a / d, b / d, c / d);
	}
	float norm() {
		return sqrtf(a * a + b * b + c * c);
	}
	Vec3f normalize() {
		return *this / norm();
	}
};

class Vec4f {
	float a, b, c, d;
	public:
	Vec4f(float aa = 0, float bb = 0, float cc = 0, float dd = 0) : a(aa), b(bb), c(cc), d(dd) {}
	const Vec4f& operator=(const Vec4f& v) {
		a = v.a;
		b = v.b;
		c = v.c;
		d = v.d;
		return *this;
	}
	float get_a() const {
		return a;
	}
	float get_b() const {
		return b;
	}
	float get_c() const {
		return c;
	}
	float get_d() const {
		return d;
	}
	Vec4f operator+(Vec4f v) const {
		return Vec4f(a + v.a, b + v.b, c + v.c, d + v.d);
	}
	Vec4f operator-(Vec4f v) const {
		return Vec4f(a - v.a, b - v.b, c - v.c, d - v.d);
	}
	float operator*(Vec4f v) const {
		return a * v.a + b * v.b + c * v.c + d * v.d;
	}
	Vec4f operator*(float m) const {
		return Vec4f(a * m, b * m, c * m, d * m);
	}
	float &operator[](int i) //перегрузка []     
    {
		if (i == 0) {
			return a;
		}
		else if (i == 1) {
			return b;
		}
		else if (i == 2) {
			return c;
		}
		else if (i == 3) {
			return d;
		}
	}
	Vec4f operator/(float n) const {
		return Vec4f(a / n, b / n, c / n, d / n);
	}
	float norm() {
		return sqrtf(a * a + b * b + c * c + d * d);
	}
	Vec4f normalize() {
		return *this / norm();
	}
};


unsigned char *image_t;
std::vector<Vec3f> envmap;

unsigned int convert(const Vec3f w){
  unsigned int r;
  Vec3f v = w;
  r = 255 * v.get_a() + int(v.get_b() * 256 *255 / 256) * 256 + int(v.get_c() *255 *256*256 / 256 / 256) *256 * 256;
  return r;
}

struct Light {
    Light(const Vec3f &p, const float &i) : position(p), intensity(i) {}
    Vec3f position;
    float intensity;
};


struct Material {
    Material(const float &r, const Vec4f &a, const Vec3f &color, const float &spec) : refractive_index(r), albedo(a), diffuse_color(color), specular_exponent(spec) {}
    Material() : refractive_index(1), albedo(1,0,0,0), diffuse_color(), specular_exponent() {}
    float refractive_index;
    Vec4f albedo;
    Vec3f diffuse_color;
    float specular_exponent;
};

Material     mirror1(1.0, Vec4f(0.01, 100.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 500);
Material black(1.0, Vec4f(0.9,  0.1, 0.0, 0.0), Vec3f(0, 0., 0.),   10.);

struct Sphere {
    Vec3f center;
    float radius;
    Material material;

    Sphere(const Vec3f &c, const float &r, const Material &m) : center(c), radius(r), material(m) {}

    bool ray_intersect(const Vec3f &orig, const Vec3f &dir, float &t0) const {
        Vec3f L = center - orig;
        float tca = L*dir;
        float d2 = L*L - tca*tca;
        if (d2 > radius*radius) return false;
        float thc = sqrtf(radius*radius - d2);
        t0       = tca - thc;
        float t1 = tca + thc;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        return true;
    }
};





Vec3f reflect(const Vec3f &I, const Vec3f &N) {
    return I - N*2.f*(I*N);
}

Vec3f refract(const Vec3f &I, const Vec3f &N, const float &refractive_index) { // Snell's law
    float cosi = - std::max(-1.f, std::min(1.f, I*N));
    float etai = 1, etat = refractive_index;
    Vec3f n = N;
    if (cosi < 0) { 
    	cosi = -cosi;
        std::swap(etai, etat);
        n = N * (-1.);
    }
    float eta = etai / etat;
    float k = 1 - eta*eta*(1 - cosi*cosi);
    return k < 0 ? Vec3f(0,0,0) : I*eta + n*(eta * cosi - sqrtf(k));
}


bool scene_intersect(const Vec3f &orig, const Vec3f &dir, const std::vector<Sphere> &spheres, Vec3f &hit, Vec3f &N, Material &material) {
    float spheres_dist = std::numeric_limits<float>::max();
    for (size_t i=0; i < spheres.size(); i++) {
        float dist_i;
        if (spheres[i].ray_intersect(orig, dir, dist_i) && dist_i < spheres_dist) {
            spheres_dist = dist_i;
            hit = orig + dir*dist_i;
            N = (hit - spheres[i].center).normalize();
            material = spheres[i].material;
        }
    }

    float checkerboard_dist = std::numeric_limits<float>::max();
    if (fabs(dir.get_b())>1e-5)  {
    	float d = -(orig.get_b()+5)/dir.get_b(); // the checkerboard plane has equation y = -4
        Vec3f pt = orig + dir*d;
        if (d>0 && d<spheres_dist) {
            checkerboard_dist = d;
            hit = pt;
            N = Vec3f(0,1,0);
            material = (int(.5*hit.get_a()+1000) + int(.5*hit.get_c())) & 1 ? mirror1 : black;
            material.diffuse_color = material.diffuse_color*.3;
            
        }
    }
    return std::min(spheres_dist, checkerboard_dist)<1000;
}

Vec3f cast_ray(const Vec3f &orig, const Vec3f &dir, const std::vector<Sphere> &spheres, const std::vector<Light> &lights, size_t depth=0) {
    Vec3f point, N;
    Material material;
    Vec3f d = dir;
    
	
    if (depth>1 || !scene_intersect(orig, dir, spheres, point, N, material)) {
    	int w   = 1920;
		int h   = 1200;
	    int index;
    	float u;
		float v;
		convert_xyz_to_cube_uv(d[0], d[1], d[2], &index, &u, &v);
		return envmap[int(u * w) + int(v * h) * w];
	}

    Vec3f reflect_dir = reflect(dir, N).normalize();
    Vec3f refract_dir = refract(dir, N, material.refractive_index).normalize();
    Vec3f reflect_orig = reflect_dir*N < 0 ? point - N*1e-3 : point + N*1e-3; // offset the original point to avoid occlusion by the object itself
    Vec3f refract_orig = refract_dir*N < 0 ? point - N*1e-3 : point + N*1e-3;
    Vec3f reflect_color = cast_ray(reflect_orig, reflect_dir, spheres, lights, depth + 1);
    Vec3f refract_color = cast_ray(refract_orig, refract_dir, spheres, lights, depth + 1);
    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (size_t i=0; i<lights.size(); i++) {
        Vec3f light_dir      = (lights[i].position - point).normalize();
        float light_distance = (lights[i].position - point).norm();

        Vec3f shadow_orig = light_dir*N < 0 ? point - N*1e-3 : point + N*1e-3; // checking if the point lies in the shadow of the lights[i]
        Vec3f shadow_pt, shadow_N;
        Material tmpmaterial;
        if (scene_intersect(shadow_orig, light_dir, spheres, shadow_pt, shadow_N, tmpmaterial) && (shadow_pt-shadow_orig).norm() < light_distance)
            continue;
        diffuse_light_intensity  += lights[i].intensity * std::max(0.f, light_dir*N);
        specular_light_intensity += powf(std::max(0.f, reflect(light_dir, N)*dir), material.specular_exponent)*lights[i].intensity;
    }
    Vec3f v  = material.diffuse_color * diffuse_light_intensity * material.albedo[0] + Vec3f(1., 1., 1.)*specular_light_intensity * material.albedo[1] + reflect_color*material.albedo[2] + refract_color*material.albedo[3];
    for (int i=0; i<3; i++) {
    	if (v[i] > 1.) {
    		v[i] = 1.;
    	}
    }
    return v;
}

void render(const std::vector<Sphere> &spheres, std::vector<uint32_t>  &image, const int width, const int height,  const std::vector<Light> &lights) {
    const int fov = M_PI/2.;
    std::vector<Vec3f> framebuffer(width*height);
    for (size_t j = 0; j<height; j++) {
        for (size_t i = 0; i<width; i++) {
            float x = (i / (float)width  - 0.5)*tan(fov/2.)*width/(float)height;
            float y = (j / (float)height - 0.5)*tan(fov/2.);
            Vec3f dir = Vec3f(x, y, -1).normalize();

            Vec3f v = cast_ray(Vec3f(0, 0, 15), dir, spheres, lights);
            float max = std::max(v[0], std::max(v[1], v[2]));
            image[i+j*width] = convert(v);
        }
    }
}

int main(int argc, const char** argv) {
    
	std::unordered_map<std::string, std::string> cmdLineParams;
	int width    = 1024;
	int height   = 1024;
	int n = 3;
	int w   = 1920;
	int h   = 1200;
	image_t = stbi_load("../t.jpg", &w, &h, &n, 0);

	envmap = std::vector<Vec3f>(w*h);
	for (int j = h-1; j>=0 ; j--) {
		for (int i = 0; i<w; i++) {
		  envmap[i+j*w] = Vec3f(image_t[(i+j*w)*3+0], image_t[(i+j*w)*3+1], image_t[(i+j*w)*3+2])  *(1/255.);
		}
	}


  for(int i=0; i<argc; i++)
  {
    std::string key(argv[i]);

    if(key.size() > 0 && key[0]=='-')
    {
      if(i != argc-1) // not last argument
      {
        cmdLineParams[key] = argv[i+1];
        i++;
      }
      else
        cmdLineParams[key] = "";
    }
  }

  std::string outFilePath = "zout.bmp";
  if(cmdLineParams.find("-out") != cmdLineParams.end())
    outFilePath = cmdLineParams["-out"];

  int sceneId = 0;
  if(cmdLineParams.find("-scene") != cmdLineParams.end())
    sceneId = atoi(cmdLineParams["-scene"].c_str());

  Vec3f color = Vec3f(0, 0, 0);
  if(sceneId == 1)
    return 0;
  else if(sceneId == 2)
    color = Vec3f(1, 1, 0);
  else if(sceneId == 3)
    return 0;

  std::vector<uint32_t> image(width*height);
  Material      ivory(1.0, Vec4f(0.5,  0.3, 0.1, 0.0), Vec3f(0.9, 0.9, 0.9),   5.);
  Material      glass(1.5, Vec4f(0.0,  0.5, 0.1, 0.8), Vec3f(0.6, 0.7, 0.8),  125.);
  Material green(1.0, Vec4f(0.9,  0.12, 0.0, 0.0), Vec3f(0.1, 0.2, 0.3),   12.);
  Material     mirror(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);
  Material      star(1.5, Vec4f(1,  0, 0.1, 0.8), Vec3f(0.8, 0.8, 0.2),  10);

  float r = 3;
  std::vector<Sphere> spheres;
  spheres.push_back(Sphere(Vec3f(5,    3 - r,   -20), 3,      ivory));
  spheres.push_back(Sphere(Vec3f(2.5,    5 - r,   -20), 1.5,      black));
  spheres.push_back(Sphere(Vec3f(6.9,    5 - r,   -19.7), 1.5,      black));
  spheres.push_back(Sphere(Vec3f(5.8,    3.7 - r,   -17.3), 0.5,      black));
  spheres.push_back(Sphere(Vec3f(4.2,    3.7 - r,   -17.3), 0.5,      black));
  spheres.push_back(Sphere(Vec3f(5,    3.3 - r,   -17), 0.2,      black));
  spheres.push_back(Sphere(Vec3f( 0,    2,   -18), 0.7,     mirror));
  spheres.push_back(Sphere(Vec3f(-3.0, -1.5, -25), 1,      glass));
  spheres.push_back(Sphere(Vec3f(-6, 0.5, -25), 1, green));

  std::vector<Light> lights;
  lights.push_back(Light(Vec3f(-25, 30, -45), 1.5));
  lights.push_back(Light(Vec3f( 30, 30, -30), 1.8));
  lights.push_back(Light(Vec3f( 10, 40,  20), 1.7));

  render(spheres, image, width, height, lights);

  SaveBMP(outFilePath.c_str(), image.data(), width, height);

  std::cout << "end." << std::endl;
  return 0;
}