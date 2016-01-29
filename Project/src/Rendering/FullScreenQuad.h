#ifndef FULLSCREENQUAD_H
#define FULLSCREENQUAD_H
#include <GL/gl3w.h>
class FullScreenQuad {
public:
	FullScreenQuad();
	~FullScreenQuad();

	GLuint getvaoID() const;
private:
	
	GLuint m_vaoID;
};
#endif //FULLSCREENQUAD_H