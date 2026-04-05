#include "planet.hpp"
#include <iostream>
#include <string>
#include <cmath>

// Vector3D implementation
Vector3D::Vector3D(double x, double y, double z) : x(x), y(y), z(z) {}

double Vector3D::getX() const { return x; }
double Vector3D::getY() const { return y; }
double Vector3D::getZ() const { return z; }

double Vector3D::magnitude() const {
    return sqrt(x * x + y * y + z * z);
}

Vector3D Vector3D::operator+(const Vector3D& other) const {
    return Vector3D(x + other.x, y + other.y, z + other.z);
}

Vector3D Vector3D::operator-(const Vector3D& other) const {
    return Vector3D(x - other.x, y - other.y, z - other.z);
}

Vector3D Vector3D::operator*(double scalar) const {
    return Vector3D(x * scalar, y * scalar, z * scalar);
}

Vector3D Vector3D::operator/(double scalar) const {
    return Vector3D(x / scalar, y / scalar, z / scalar);
}

// Planet implementation
Planet::Planet(std::string name, double radius, double mass, Vector3D position) 
    : name(name), radius(radius), mass(mass), position(position) {}

std::string Planet::getName() const { return name; }
double Planet::getRadius() const { return radius; }
double Planet::getMass() const { return mass; }
Vector3D Planet::getPosition() const { return position; }
void Planet::setPosition(Vector3D position) { this->position = position; }

// Velocity implementation
Velocity::Velocity(Vector3D velocity) : velocity(velocity) {}
Vector3D Velocity::getVelocity() const { return velocity; }
void Velocity::setVelocity(Vector3D velocity) { this->velocity = velocity; }

// Force implementation
Force::Force(Vector3D force) : force(force) {}
Vector3D Force::getForce() const { return force; }
void Force::setForce(Vector3D force) { this->force = force; }

// Body implementation
Body::Body(Planet planet, Velocity velocity) 
    : planet(planet), velocity(velocity), force(Force(Vector3D(0, 0, 0))) {}

Planet& Body::getPlanet() { return planet; }
const Planet& Body::getPlanet() const { return planet; }
Velocity& Body::getVelocity() { return velocity; }
const Velocity& Body::getVelocity() const { return velocity; }
Force& Body::getForce() { return force; }
const Force& Body::getForce() const { return force; }
void Body::setForce(Force force) { this->force = force; }

// ThreeBodySimulation implementation
ThreeBodySimulation::ThreeBodySimulation(Body& body1, Body& body2, Body& body3, double dt)
    : body1(body1), body2(body2), body3(body3), dt(dt) {}

void ThreeBodySimulation::calculateForces() {
    auto pairForce = [&](const Body& source, const Body& other) {
        Vector3D r = other.getPlanet().getPosition() - source.getPlanet().getPosition();
        double d = r.magnitude();
        if (d == 0.0) {
            return Vector3D(0, 0, 0);
        }
        return r * (G * source.getPlanet().getMass() * other.getPlanet().getMass() / (d * d * d));
    };

    Vector3D F1 = pairForce(body1, body2) + pairForce(body1, body3);
    Vector3D F2 = pairForce(body2, body1) + pairForce(body2, body3);
    Vector3D F3 = pairForce(body3, body1) + pairForce(body3, body2);

    body1.setForce(Force(F1));
    body2.setForce(Force(F2));
    body3.setForce(Force(F3));
}

void ThreeBodySimulation::applyVelocityKick(double stepDt) {
    // Velocity-Verlet kick: v = v + a * dt_segment
    Vector3D v1 = body1.getVelocity().getVelocity() + 
                  body1.getForce().getForce() / body1.getPlanet().getMass() * stepDt;
    body1.getVelocity().setVelocity(v1);
    
    Vector3D v2 = body2.getVelocity().getVelocity() + 
                  body2.getForce().getForce() / body2.getPlanet().getMass() * stepDt;
    body2.getVelocity().setVelocity(v2);
    
    Vector3D v3 = body3.getVelocity().getVelocity() + 
                  body3.getForce().getForce() / body3.getPlanet().getMass() * stepDt;
    body3.getVelocity().setVelocity(v3);
}

void ThreeBodySimulation::updatePositions() {
    // Drift using the half-step velocity from velocity-Verlet.
    body1.getPlanet().setPosition(body1.getPlanet().getPosition() + 
                                   body1.getVelocity().getVelocity() * dt);
    body2.getPlanet().setPosition(body2.getPlanet().getPosition() + 
                                   body2.getVelocity().getVelocity() * dt);
    body3.getPlanet().setPosition(body3.getPlanet().getPosition() + 
                                   body3.getVelocity().getVelocity() * dt);
}

void ThreeBodySimulation::step() {
    calculateForces();
    applyVelocityKick(0.5 * dt);
    updatePositions();
    calculateForces();
    applyVelocityKick(0.5 * dt);
}
