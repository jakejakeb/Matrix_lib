#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <utility>

template <typename Type>
class Mat2D
{
public:
	size_t _cols, _rows;
	std::vector<Type> _vals;

	Mat2D(size_t cols, size_t rows) : _cols(cols), _rows(rows), _vals({})
	{
		_vals.resize(rows * cols, Type());
	}
	Mat2D() : _cols(0), _rows(0), _vals({}) {}

	Type& at(size_t col, size_t row)
	{
		return _vals[row * _cols + col];
	}

	bool isSquare()
	{
		return _rows == _cols;
	}

	Mat2D negetive()
	{
		Mat2D output(_cols, _rows);
		for (size_t y = 0; y < output._rows; y++)
			for (size_t x = 0; x < output._cols; x++)
			{
				output.at(x, y) = -at(x, y);
			}
		return output;
	}

	Mat2D mult(Mat2D& target)
	{
		assert(_cols == target._rows);
		Mat2D output(target._cols, _rows);
		for (size_t y = 0; y < output._rows; y++)
			for (size_t x = 0; x < output._cols; x++)
			{
				Type result = Type();
				for (size_t k = 0; k < _cols; k++)
					result += at(k, y) * target.at(x, k);
				output.at(x, y) = result;
			}
		return output;
	}

	Mat2D multElem(Mat2D& target)
	{
		assert(_rows == target._rows && _cols == target._cols);
		Mat2D output(_cols, _rows);
		for (size_t y = 0; y < output._rows; y++)
			for (size_t x = 0; x < output._cols; x++)
			{
				output.at(x, y) = at(x, y) * target.at(x, y);
			}
		return output;
	}

	Mat2D add(Mat2D& target)
	{
		assert(_rows == target._rows && _cols == target._cols);
		Mat2D output(_cols, _rows);
		for (size_t y = 0; y < output._rows; y++)
			for (size_t x = 0; x < output._cols; x++)
			{
				output.at(x, y) = at(x, y) + target.at(x, y);
			}
		return output;
	}

	Mat2D applyFunction(std::function<Type(const Type&)> func)
	{
		Mat2D output(_cols, _rows);
		for (size_t y = 0; y < output._rows; y++)
			for (size_t x = 0; x < output._cols; x++)
			{
				output.at(x, y) = func(at(x, y));
			}
		return output;
	}

	Mat2D multScaler(float s)
	{
		Mat2D output(_cols, _rows);
		for (size_t y = 0; y < output._rows; y++)
			for (size_t x = 0; x < output._cols; x++)
			{
				output.at(x, y) = at(x, y) * s;
			}
		return output;
	}

	Mat2D addScaler(float s)
	{
		Mat2D output(_cols, _rows);
		for (size_t y = 0; y < output._rows; y++)
			for (size_t x = 0; x < output._cols; x++)
			{
				output.at(x, y) = at(x, y) + s;
			}
		return output;
	}

	Mat2D transpose()
	{
		Mat2D output(_rows, _cols);
		for (size_t y = 0; y < _rows; y++)
			for (size_t x = 0; x < _cols; x++)
			{
				output.at(y, x) = at(x, y);
			}
		return output;
	}

	Mat2D cofactor(size_t col, size_t row)
	{
		Mat2D output(_cols - 1, _rows - 1);
		size_t i = 0;
		for (size_t y = 0; y < _rows; y++)
			for (size_t x = 0; x < _cols; x++)
			{
				if (x == col || y == row) continue;
				output._vals[i++] = at(x, y);
			}
		return output;
	}

	Type determinant()
	{
		assert(_rows == _cols);
		Type output = Type();

		if (_rows == 1)
			return _vals[0];
		else
		{
			int sign = 1;
			for (size_t x = 0; x < _cols; x++)
			{
				output += sign * at(x, 0) * cofactor(x, 0).determinant();
				sign *= -1;
			}
		}
		return output;
	}

	Mat2D adjoint()
	{
		assert(_rows == _cols);
		Mat2D output(_cols, _rows);
		int sign = 1;
		for (size_t y = 0; y < _rows; y++)
			for (size_t x = 0; x < _cols; x++)
			{
				output.at(x, y) = sign * cofactor(x, y).determinant();
				sign *= -1;
			}
		output = output.transpose();

		return output;
	}

	Mat2D inverse()
	{
		Mat2D adj = adjoint();
		Type factor = determinant();
		for (size_t y = 0; y < adj._cols; y++)
			for (size_t x = 0; x < adj._rows; x++)
			{
				adj.at(x, y) = adj.at(x, y) / factor;
			}
		return adj;
	}

	void print() {
		for (size_t y = 0; y < _rows; y++) {
			for (size_t x = 0; x < _cols; x++)
				std::cout << std::setw(1) << at(x, y) << " ";
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
};