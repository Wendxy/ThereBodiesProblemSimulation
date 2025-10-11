#ifndef PLANET_HPP
#define PLANET_HPP
#include <iostream>
#include <string>
#include <cmath>

// Forward declarations
class Body;

class Vector3D{
    public:
        Vector3D(double x = 0, double y = 0, double z = 0);
        double getX() const;
        double getY() const;
        double getZ() const;
        double magnitude() const;
        Vector3D operator+(const Vector3D& other) const;
        Vector3D operator-(const Vector3D& other) const;
        Vector3D operator*(double scalar) const;
        Vector3D operator/(double scalar) const;
    private:
        double x;
        double y;
        double z;
};

class Planet {
    public:
        Planet(std::string name, double radius, double mass, Vector3D position);
        std::string getName() const;
        double getRadius() const;
        double getMass() const;
        Vector3D getPosition() const;
        void setPosition(Vector3D position);
    private:
        std::string name;
        double radius;
        double mass;
        Vector3D position;
};

class Velocity {
    public:
        Velocity(Vector3D velocity);
        Vector3D getVelocity() const;
        void setVelocity(Vector3D velocity);
    private:
        Vector3D velocity;
};

class Force {
    public:
        Force(Vector3D force);
        Vector3D getForce() const;
        void setForce(Vector3D force);
    private:
        Vector3D force;
};

class Body {
    public:
        Body(Planet planet, Velocity velocity);
        Planet& getPlanet();
        const Planet& getPlanet() const;
        Velocity& getVelocity();
        const Velocity& getVelocity() const;
        Force& getForce();
        const Force& getForce() const;
        void setForce(Force force);
    private:
        Planet planet;
        Velocity velocity;
        Force force;
};

class ThreeBodySimulation {
    public:
        ThreeBodySimulation(Body& body1, Body& body2, Body& body3, double dt);
        void calculateForces();
        void updateVelocities();
        void updatePositions();
        void step();
    private:
        Body& body1;
        Body& body2;
        Body& body3;
        double dt;  // time step
        const double G = 6.67430e-11;
};

#endif // PLANET_HPP

