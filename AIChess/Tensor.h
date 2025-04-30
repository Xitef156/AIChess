#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>

template <typename T>
class Tensor
{
public:
    // Dimensions du tenseur
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::vector<T> data;

    // Constructeurs
    Tensor() = default;

    explicit Tensor(const std::vector<size_t>& shape)
    {
        resize(shape);
    }

    Tensor(const std::vector<size_t>& shape, const std::vector<T>& data)
        : shape(shape), data(data)
    {
        if (totalSize() != data.size())
        {
            throw std::invalid_argument("Les données ne correspondent pas à la taille spécifiée par la forme.");
        }
        calculateStrides();
    }

    void resize(const std::vector<size_t>& new_shape)
    {
        shape = new_shape;
        data.resize(totalSize());
        calculateStrides();
    }

    size_t totalSize() const
    {
        size_t size = 1;
        for (size_t dim : shape)
        {
            size *= dim;
        }
        return size;
    }

    // Accès aux éléments
    T& at(const std::vector<size_t>& indices)
    {
        validateIndices(indices);
        return data[toLinearIndex(indices)];
    }

    const T& at(const std::vector<size_t>& indices) const
    {
        validateIndices(indices);
        return data[toLinearIndex(indices)];
    }

    // Opérations mathématiques élément par élément
    Tensor operator+(const Tensor& other) const
    {
        checkShapeCompatibility(other, "addition");
        return combine(other, [](T a, T b) { return a + b; });
    }

    Tensor operator-(const Tensor& other) const
    {
        checkShapeCompatibility(other, "soustraction");
        return combine(other, [](T a, T b) { return a - b; });
    }

    Tensor operator*(const T& scalar) const
    {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    Tensor operator/(const T& scalar) const
    {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] / scalar;
        }
        return result;
    }

    Tensor apply(const std::function<T(T)>& func) const
    {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = func(data[i]);
        }
        return result;
    }

    // Produit matriciel
    Tensor matmul(const Tensor& other) const
    {
        if (shape.size() != 2 || other.shape.size() != 2 || shape[1] != other.shape[0])
        {
            throw std::invalid_argument("Dimensions incompatibles pour le produit matriciel.");
        }
        Tensor result({ shape[0], other.shape[1] });
        for (size_t i = 0; i < shape[0]; ++i)
        {
            for (size_t j = 0; j < other.shape[1]; ++j)
            {
                T sum = 0;
                for (size_t k = 0; k < shape[1]; ++k)
                {
                    sum += at({ i, k }) * other.at({ k, j });
                }
                result.at({ i, j }) = sum;
            }
        }
        return result;
    }

    // Sauvegarde et chargement
    void saveToStream(std::ostream& out) const
    {
        size_t shape_size = shape.size();
        out.write(reinterpret_cast<const char*>(&shape_size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(shape.data()), shape_size * sizeof(size_t));
        out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));

        if (!out)
            throw std::runtime_error("Erreur lors de l'écriture dans le flux.");
    }

    void loadFromStream(std::istream& in)
    {
        size_t shape_size;
        in.read(reinterpret_cast<char*>(&shape_size), sizeof(size_t));

        if (!in)
            throw std::runtime_error("Erreur lors de la lecture de la taille de la forme.");

        shape.resize(shape_size);
        in.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(size_t));

        if (!in)
            throw std::runtime_error("Erreur lors de la lecture des dimensions.");

        size_t total_size = totalSize();
        data.resize(total_size);
        in.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(T));

        if (!in)
            throw std::runtime_error("Erreur lors du chargement des données du tenseur.");

        calculateStrides();
    }

    void calculateStrides()
    {
        strides.resize(shape.size());
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    size_t toLinearIndex(const std::vector<size_t>& indices) const
    {
        size_t linear_index = 0;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            linear_index += indices[i] * strides[i];
        }
        return linear_index;
    }

    void validateIndices(const std::vector<size_t>& indices) const
    {
        if (indices.size() != shape.size())
        {
            throw std::out_of_range("Le nombre d'indices ne correspond pas au nombre de dimensions.");
        }
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] >= shape[i])
            {
                std::cout << indices[i] << " ; " << shape[i] << std::endl;
                throw std::out_of_range("Indice hors des limites pour la dimension " + std::to_string(i));
            }
        }
    }

    void checkShapeCompatibility(const Tensor& other, const std::string& operation_name) const
    {
        if (shape != other.shape)
        {
            throw std::invalid_argument("Les formes des tenseurs ne sont pas compatibles pour " + operation_name + ".");
        }
    }

    Tensor combine(const Tensor& other, const std::function<T(T, T)>& func) const
    {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = func(data[i], other.data[i]);
        }
        return result;
    }

    Tensor operator*(const Tensor& other) const
    {
        checkShapeCompatibility(other, "multiplication élément par élément");
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i)
        {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }
};
