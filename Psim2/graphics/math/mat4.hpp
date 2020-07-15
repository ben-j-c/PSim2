#ifndef GRAPHICSmat4
#define GRAPHICSmat4
#include "vec4.hpp"

namespace Graphics{
	class mat4location;

	class mat4 {
		friend class mat4location;
		friend class vec4;
	public:
		static const mat4 Identity;
		mat4(bool identity = false);
		mat4(const mat4&);
		mat4(const vec4&, const vec4&, const vec4&, const vec4&);
		mat4(float values[4][4]);
		mat4(float values[16]);
		mat4(float);
		~mat4();

		mat4 operator+(const mat4&);
		mat4 operator-(const mat4 & add);
		mat4& operator+=(const mat4&);
		mat4 & operator-=(const mat4 & add);
		mat4 operator*(const mat4&);
		mat4 operator*(const float values[16]);
		mat4 operator*(const float values[4][4]);
		vec4 operator*(const vec4&);
		mat4& operator*=(const mat4&);
		mat4& operator=(const mat4&);

		mat4 operator+(float);
		mat4& operator+=(float);
		mat4 operator*(float);
		mat4& operator*=(float);
		mat4& operator=(float);

		mat4location operator[](int index);
		mat4location operator()(int row, int col);
		inline const float at(int row, int col);
		inline const void set(int row, int col, float setValue);

		float* getCopy();
		float* data();

	private:
		//int index;
		float value[4][4] = {0};
	};
	class mat4location {
		friend class mat4;
	public:
		inline operator float() const {return matrix->value[row][col];} //right hand
		inline void operator=(float newValue){ //left hand
			matrix->value[row][col] = newValue;
		}

		inline void operator+=(float newValue) { //left hand
			matrix->value[row][col] += newValue;
		}

		inline void operator-=(float newValue) { //left hand
			matrix->value[row][col] -= newValue;
		}

		inline void operator=(const mat4location& newValue){ //left hand
			matrix->value[row][col] = newValue.matrix->value[newValue.row][newValue.col];
		}
		mat4location(mat4* matrix,int row, int col) {
			this->matrix = matrix;
			this->row = row;
			this->col = col;
		}
		int row;
		int col;
		mat4* matrix;
	};

	mat4 getIdentity();
	void getPerspectiveProjection(float);
	void printMat4(const mat4&);
}



#endif