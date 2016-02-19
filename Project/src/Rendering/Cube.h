#ifndef CUBE_H
#define CUBE_H

#include <GL/gl3w.h>

/** \addtogroup Rendering
* {@
*/

	/**
	 * @brief The Cube is a class that inherits from Renderable.
	 * @detaisl As the name implies, it is indeed a cube. This Renderable does not
	 * contain any information about material. So please set manually if needed.
	*/
	struct VertexBufferObject
	{
		GLuint m_vboHandle; //!< A VertexBufferObject handle
		GLuint m_size; //!< the size of a VertexBufferObject.
	};

	class Cube
	{
	public:
		/**
		 * @brief Create a Cube Renderable
		 * 
		 * @param size edge length of the cube
		 */
		Cube(float size = 1.0f);
		~Cube();

		void render();

	private:
		GLenum m_mode; //!< the mode the Renderable will be drawn with (e.g. GL_TRIANGLES)
		GLuint m_vao; //!< VertexArrayObject handle
		VertexBufferObject m_indices; //!< index buffer
		VertexBufferObject m_positions; //!< position buffer
		VertexBufferObject m_normals; //!< normal buffer
		VertexBufferObject m_uvs; //!< uv buffer
		VertexBufferObject m_tangents; //!< tangent buffer
		VertexBufferObject m_bitangents; //!< tangent buffer
	};


/** @}*/
#endif // CUBE_H
