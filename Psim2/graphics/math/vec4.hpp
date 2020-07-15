#ifndef GRAPHICSvec4
#define GRAPHICSvec4
#include <string>

namespace Graphics{
	class vec4location;
	class mat4;
	class vec4{
		friend class mat4;
		friend class vec4location;
	public:
		vec4();
		vec4(float, float, float, float);
		vec4(float newValue[4]);
		vec4(float);

		vec4& operator=(const vec4&);

		vec4 operator+(const vec4&);
		vec4 operator-(const vec4&);
		float operator*(const vec4&);
		float dot(const vec4&);
		vec4 cross(const vec4&);
		vec4& operator+=(const vec4&);
		vec4& operator-=(const vec4&);
		vec4& operator*=(const vec4&);

		vec4 operator+(float);
		vec4 operator-(float);
		vec4 operator*(float);
		vec4 operator/(float);
		vec4& operator+=(float);
		vec4& operator-=(float);
		vec4& operator*=(float);
		vec4& operator/=(float);

		vec4 operator||(const vec4&);
		vec4 projectOnto(const vec4&);
		vec4 operator^(const vec4&);
		vec4 projectOrthogonal(const vec4&);

		vec4& unitize();
		vec4 unit();
		float magnitude();
		float mag();
		float magnitudeSquared();
		float magSq();

		vec4& rotateX(float);
		vec4& rotateY(float);
		vec4& rotateZ(float);

		vec4location operator[](int);
		vec4location operator()(int);

		operator std::string ();

		inline const float at(int row) {return value[row];};
		inline const float set(int row, float setValue) {return value[row] = setValue;};
	private:
		float value[4] = {0};
	};

	class vec4location {
		friend class vec4;
	public:
		inline operator float() const {return vect->value[row];} //right hand
		inline void operator=(float newValue){ //left hand
			vect->value[row] = newValue;
		}

		inline void operator=(const vec4location& newValue){ //left hand
			vect->value[row] = newValue.vect->value[newValue.row];
		}
		vec4location(vec4* vect,int row) {
			this->vect = vect;
			this->row = row;
		}
		int row;
		vec4* vect;
	};
}

#endif