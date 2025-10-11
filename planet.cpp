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
    // Get positions
    Vector3D r1 = body1.getPlanet().getPosition();
    Vector3D r2 = body2.getPlanet().getPosition();
    Vector3D r3 = body3.getPlanet().getPosition();
    
    // Get masses
    double m1 = body1.getPlanet().getMass();
    double m2 = body2.getPlanet().getMass();
    double m3 = body3.getPlanet().getMass();
    
    // Calculate force on body1
    Vector3D r12 = r2 - r1;
    Vector3D r13 = r3 - r1;
    double d12 = r12.magnitude();
    double d13 = r13.magnitude();
    Vector3D F1 = r12 * (G * m1 * m2 / (d12 * d12 * d12)) + 
                  r13 * (G * m1 * m3 / (d13 * d13 * d13));
    
    // Calculate force on body2
    Vector3D r21 = r1 - r2;
    Vector3D r23 = r3 - r2;
    double d21 = r21.magnitude();
    double d23 = r23.magnitude();
    Vector3D F2 = r21 * (G * m2 * m1 / (d21 * d21 * d21)) + 
                  r23 * (G * m2 * m3 / (d23 * d23 * d23));
    
    // Calculate force on body3
    Vector3D r31 = r1 - r3;
    Vector3D r32 = r2 - r3;
    double d31 = r31.magnitude();
    double d32 = r32.magnitude();
    Vector3D F3 = r31 * (G * m3 * m1 / (d31 * d31 * d31)) + 
                  r32 * (G * m3 * m2 / (d32 * d32 * d32));
    
    // Set forces
    body1.setForce(Force(F1));
    body2.setForce(Force(F2));
    body3.setForce(Force(F3));
}

void ThreeBodySimulation::updateVelocities() {
    // v = v + a*dt, where a = F/m
    Vector3D v1 = body1.getVelocity().getVelocity() + 
                  body1.getForce().getForce() / body1.getPlanet().getMass() * dt;
    body1.getVelocity().setVelocity(v1);
    
    Vector3D v2 = body2.getVelocity().getVelocity() + 
                  body2.getForce().getForce() / body2.getPlanet().getMass() * dt;
    body2.getVelocity().setVelocity(v2);
    
    Vector3D v3 = body3.getVelocity().getVelocity() + 
                  body3.getForce().getForce() / body3.getPlanet().getMass() * dt;
    body3.getVelocity().setVelocity(v3);
}

void ThreeBodySimulation::updatePositions() {
    // r = r + v*dt
    body1.getPlanet().setPosition(body1.getPlanet().getPosition() + 
                                   body1.getVelocity().getVelocity() * dt);
    body2.getPlanet().setPosition(body2.getPlanet().getPosition() + 
                                   body2.getVelocity().getVelocity() * dt);
    body3.getPlanet().setPosition(body3.getPlanet().getPosition() + 
                                   body3.getVelocity().getVelocity() * dt);
}

void ThreeBodySimulation::step() {
    calculateForces();
    updateVelocities();
    updatePositions();
}